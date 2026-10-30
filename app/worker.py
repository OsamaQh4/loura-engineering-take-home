"""Background classification workers.

N asyncio tasks run inside the API process. Each loops: claim a due ticket,
call the model with a timeout, validate, then either store the result,
schedule a retry with backoff, or mark the ticket failed.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable

from app.classifier import InvalidModelOutput, build_prompt, parse_classification
from app.config import Settings
from app.llm import LLMClient, LLMError
from app.store import Claim, TicketStore

log = logging.getLogger(__name__)


class WorkerPool:
    def __init__(
        self,
        store: TicketStore,
        llm: LLMClient,
        settings: Settings,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.llm = llm
        self.settings = settings
        self.clock = clock
        self._wake = asyncio.Event()
        self._stopping = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._loop(), name=f"classifier-{i}")
            for i in range(self.settings.worker_concurrency)
        ]

    def notify(self) -> None:
        """Wake idle workers (called after a ticket is created)."""
        self._wake.set()

    async def stop(self) -> None:
        """Stop claiming, let in-flight calls finish within the grace period, then cancel.

        A cancelled worker hands its ticket back to the queue (see `process`),
        so nothing is lost and nothing is processed twice.
        """
        self._stopping = True
        self._wake.set()
        if not self._tasks:
            return
        _, still_running = await asyncio.wait(self._tasks, timeout=self.settings.shutdown_grace_s)
        for task in still_running:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                did_work = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # never let one bad ticket kill the worker
                log.exception("worker iteration failed")
                did_work = False
            if not did_work and not self._stopping:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), self.settings.poll_interval_s)
                except asyncio.TimeoutError:
                    pass

    # ----------------------------------------------------------------- work

    async def run_once(self) -> bool:
        """Claim and process one ticket. Returns False if nothing was due."""
        claim = await asyncio.to_thread(self.store.claim_next, self.clock(), self.settings.lease_s)
        if claim is None:
            return False
        await self.process(claim)
        return True

    async def process(self, claim: Claim) -> None:
        s = self.settings
        if claim.attempts > s.max_attempts:
            # Only reachable if earlier attempts were lost to crashes (expired leases).
            # This bounds tickets that repeatedly take the worker down with them.
            await self._finish_failed(claim, f"gave up after {s.max_attempts} attempts (last attempt was interrupted)")
            return

        system, user = build_prompt(claim.subject, claim.body)
        try:
            raw = await asyncio.wait_for(self.llm.complete(system, user), s.llm_timeout_s)
            result = parse_classification(raw)
        except asyncio.CancelledError:
            # Shutdown: return the ticket to the queue immediately. Synchronous on
            # purpose, so a second cancellation can't interrupt the hand-back.
            self.store.release(claim, self.clock())
            raise
        except asyncio.TimeoutError:
            error = f"model call timed out after {s.llm_timeout_s:g}s"
        except LLMError as exc:
            error = f"model call failed: {exc}"
        except InvalidModelOutput as exc:
            error = f"invalid model output: {exc}"
            log.warning("ticket %s: %s; raw=%r", claim.ticket_id, exc, raw[:500])
        except Exception as exc:
            error = f"unexpected error: {type(exc).__name__}"
            log.exception("ticket %s: unexpected error", claim.ticket_id)
        else:
            stored = await asyncio.to_thread(self.store.complete, claim, result, self.clock())
            if not stored:
                log.warning("ticket %s: lease lost before completion; result discarded", claim.ticket_id)
            return

        if claim.attempts >= s.max_attempts:
            await self._finish_failed(claim, error)
        else:
            delay = self._backoff(claim.attempts)
            log.info("ticket %s: attempt %d failed (%s); retrying in %.1fs", claim.ticket_id, claim.attempts, error, delay)
            await asyncio.to_thread(self.store.schedule_retry, claim, error, self.clock() + delay, self.clock())

    async def _finish_failed(self, claim: Claim, error: str) -> None:
        log.warning("ticket %s: failed permanently: %s", claim.ticket_id, error)
        await asyncio.to_thread(self.store.fail, claim, error, self.clock())

    def _backoff(self, attempt: int) -> float:
        s = self.settings
        delay = min(s.retry_max_delay_s, s.retry_base_delay_s * 2 ** (attempt - 1))
        return delay * random.uniform(0.5, 1.0)  # jitter, so retries don't arrive in lockstep

"""Ticket lifecycle under an unreliable model: retries, failure, crashes, shutdown, concurrency."""

import asyncio
import json

import pytest

from app.llm import FakeLLM, LLMError
from app.models import Status
from app.store import TicketStore
from app.worker import WorkerPool
from tests.conftest import GOOD, SAMPLES, ScriptedLLM, ticket


def run(coro):
    return asyncio.run(coro)


async def drain(pool: WorkerPool) -> None:
    """Process until nothing is due (retry delays are 0 in tests)."""
    while await pool.run_once():
        pass


def test_valid_output_is_stored(store, settings):
    llm = ScriptedLLM(GOOD)
    store.create(ticket(1), 0)
    run(drain(WorkerPool(store, llm, settings)))
    t = store.get("T-1")
    assert t.status is Status.classified
    assert t.classification.category.value == "billing"
    assert (t.attempts, llm.calls, t.last_error) == (1, 1, None)


def test_invalid_output_is_retried_then_succeeds(store, settings):
    llm = ScriptedLLM('{"category": "refunds", "priority": "high", "summary": "x"}', LLMError("503"), "HANG", GOOD)
    settings = settings.__class__(**{**settings.__dict__, "max_attempts": 4})
    store.create(ticket(1), 0)
    run(drain(WorkerPool(store, llm, settings)))
    t = store.get("T-1")
    assert t.status is Status.classified
    assert t.attempts == 4 and t.last_error is None


def test_persistently_bad_output_ends_in_failed_and_is_never_stored(store, settings):
    llm = ScriptedLLM('{"category": "billing", "priority": "urgent", "summary": "x"}')
    store.create(ticket(1), 0)
    run(drain(WorkerPool(store, llm, settings)))
    t = store.get("T-1")
    assert t.status is Status.failed
    assert t.classification is None
    assert llm.calls == settings.max_attempts
    assert "invalid model output" in t.last_error and "priority" in t.last_error


@pytest.mark.parametrize(
    "response, expected_error",
    [("HANG", "timed out after 0.2s"), (LLMError("rate limited"), "model call failed: rate limited")],
)
def test_timeouts_and_provider_errors_end_in_failed_with_reason(store, settings, response, expected_error):
    store.create(ticket(1), 0)
    run(drain(WorkerPool(store, ScriptedLLM(response), settings)))
    t = store.get("T-1")
    assert t.status is Status.failed and t.attempts == settings.max_attempts
    assert expected_error in t.last_error


def test_ticket_that_keeps_crashing_workers_eventually_fails(store, settings):
    """Each crash leaves an expired lease; after max_attempts we stop trying without calling the model."""
    store.create(ticket(1), 0)
    for i in range(settings.max_attempts):
        store.claim_next(now=i * 100, lease_s=60)  # claimed, then the "worker" dies
    llm = ScriptedLLM(GOOD)
    pool = WorkerPool(store, llm, settings, clock=lambda: 10_000)
    run(pool.run_once())
    assert store.get("T-1").status is Status.failed
    assert llm.calls == 0


def test_pending_tickets_from_a_previous_run_are_processed_on_start(store, settings):
    store.create(ticket(1), 0)
    store.claim_next(now=0, lease_s=1)  # in flight when the old process died

    async def main():
        pool = WorkerPool(TicketStore(settings.database_path), ScriptedLLM(GOOD), settings)
        pool.start()
        await wait_for_status(store, "T-1", Status.classified)
        await pool.stop()

    run(main())
    assert store.get("T-1").attempts == 2


def test_graceful_shutdown_returns_in_flight_ticket_to_queue(store, settings):
    store.create(ticket(1), 0)
    settings = settings.__class__(**{**settings.__dict__, "llm_timeout_s": 60})

    async def main():
        llm = ScriptedLLM("HANG")
        pool = WorkerPool(store, llm, settings)
        pool.start()
        while llm.calls == 0:
            await asyncio.sleep(0.01)
        await pool.stop()  # grace period (0.2s) expires, worker is cancelled

    run(main())
    t = store.get("T-1")
    assert t.status is Status.pending and t.attempts == 0
    # Immediately claimable again: no waiting for the lease to expire.
    assert store.claim_next(now=0, lease_s=60) is not None


def test_concurrency_is_bounded(store, settings):
    active = peak = 0

    class SlowLLM:
        async def complete(self, system, user):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return GOOD

    for i in range(10):
        store.create(ticket(i), 0)

    async def main():
        pool = WorkerPool(store, SlowLLM(), settings)
        pool.start()
        for i in range(10):
            await wait_for_status(store, f"T-{i}", Status.classified)
        await pool.stop()

    run(main())
    assert peak == settings.worker_concurrency


def test_samples_with_flaky_fake_all_reach_a_valid_terminal_state(store, settings):
    """Every sample, including the injection attempt, ends classified-with-valid-values or failed."""
    from app.models import TicketIn

    for raw in json.loads(SAMPLES.read_text()):
        store.create(TicketIn(**raw), 0)
    llm = FakeLLM(failure_rate=0.5, seed=7, latency_s=(0, 0))
    run(drain(WorkerPool(store, llm, settings)))

    items, _ = store.list(limit=200)
    assert len(items) == len(json.loads(SAMPLES.read_text()))
    for t in items:
        assert t.status in (Status.classified, Status.failed)
        assert (t.classification is not None) == (t.status is Status.classified)
    injected = store.get("T-1007")
    if injected.classification:
        assert "refund of $5,000" not in injected.classification.summary


async def wait_for_status(store, ticket_id, status, timeout=5.0):
    async def poll():
        while store.get(ticket_id).status is not status:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)

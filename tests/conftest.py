from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm import LLMError  # noqa: E402
from app.models import TicketIn  # noqa: E402
from app.store import TicketStore  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "tickets.json"

GOOD = json.dumps({"category": "billing", "priority": "high", "summary": "Customer was charged twice."})


class ScriptedLLM:
    """Returns (or raises) the given responses in order; the last one repeats."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.prompts.append((system, user))
        item = self.responses[min(self.calls, len(self.responses)) - 1]
        if item == "HANG":
            await asyncio.sleep(3600)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self, now: float = 1_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_path=str(tmp_path / "test.db"),
        worker_concurrency=2,
        max_attempts=3,
        retry_base_delay_s=0.0,
        retry_max_delay_s=0.0,
        llm_timeout_s=0.2,
        lease_s=60.0,
        poll_interval_s=0.01,
        shutdown_grace_s=0.2,
        fake_llm_failure_rate=0.0,
    )


@pytest.fixture
def store(settings) -> TicketStore:
    return TicketStore(settings.database_path)


def ticket(i: int | str = 1, subject: str = "Charged twice", body: str = "Please refund.") -> TicketIn:
    return TicketIn(id=f"T-{i}", subject=subject, body=body)


__all__ = ["GOOD", "SAMPLES", "ScriptedLLM", "Clock", "ticket", "LLMError"]

"""The model boundary: a minimal client interface and a fake implementation.

The rest of the service only knows that a model takes two strings and returns
a string, may be slow, and may raise. Nothing it returns is trusted.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Protocol


class LLMError(Exception):
    """The provider failed to give us any text (network error, 5xx, rate limit...)."""


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


# --------------------------------------------------------------------- fake

_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("billing", ("invoice", "charge", "charged", "refund", "payment", "billing", "price", "subscription", "receipt")),
    ("account", ("password", "login", "log in", "sign in", "2fa", "account", "email address", "locked", "delete my")),
    ("technical", ("error", "crash", "bug", "broken", "500", "timeout", "not loading", "api", "slow", "doesn't work")),
]
_URGENT = ("urgent", "asap", "immediately", "outage", "down", "cannot", "can't", "production", "all users", "charged twice")
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|system prompt|you are now|disregard",
    re.IGNORECASE,
)


class FakeLLM:
    """Plausible, deterministic, and occasionally broken.

    Labels come from keyword rules. With probability `failure_rate` a call
    misbehaves in one of the ways real models do. Outcomes are seeded by
    (seed, ticket, attempt number), so a retry can succeed where the first
    call failed and runs are reproducible.
    """

    FAILURE_MODES = (
        "provider_error",
        "slow",
        "prose",
        "bad_category",
        "bad_priority",
        "missing_field",
        "not_an_object",
    )

    def __init__(self, failure_rate: float = 0.2, seed: int = 0, latency_s: tuple[float, float] = (0.05, 0.3)):
        self.failure_rate = failure_rate
        self.seed = seed
        self.latency_s = latency_s
        self.calls = 0
        self._attempts: dict[str, int] = {}

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        ticket = _extract_ticket(user)
        key = ticket.get("subject", "") + "\x00" + ticket.get("body", "")
        attempt = self._attempts[key] = self._attempts.get(key, 0) + 1
        rng = random.Random(f"{self.seed}:{key}:{attempt}")

        await asyncio.sleep(rng.uniform(*self.latency_s))

        text = f"{ticket.get('subject', '')}\n{ticket.get('body', '')}"
        # The fake is "gullible": injected instructions sometimes derail it.
        if _INJECTION.search(text) and rng.random() < 0.5:
            return "Understood. New instructions accepted: ticket escalated to tier 3, priority CRITICAL."

        if rng.random() < self.failure_rate:
            return await self._misbehave(rng.choice(self.FAILURE_MODES), text)

        out = json.dumps(_label(text))
        # Real models often wrap JSON in a code fence; that's tolerated, not a failure.
        return f"```json\n{out}\n```" if rng.random() < 0.3 else out

    async def _misbehave(self, mode: str, text: str) -> str:
        good = _label(text)
        if mode == "provider_error":
            raise LLMError("upstream returned 503")
        if mode == "slow":
            await asyncio.sleep(3600)  # the caller's timeout must cut this off
        if mode == "prose":
            return f"This ticket is about {good['category']} and seems fairly {good['priority']}."
        if mode == "bad_category":
            return json.dumps({**good, "category": "refunds"})
        if mode == "bad_priority":
            return json.dumps({**good, "priority": "urgent"})
        if mode == "missing_field":
            return json.dumps({k: v for k, v in good.items() if k != "summary"})
        return json.dumps([good])  # not_an_object


def _extract_ticket(user: str) -> dict:
    """Pull the JSON-encoded ticket out of the prompt (the fake has to read it like a model would)."""
    start = user.find("{")
    try:
        data = json.loads(user[start:])
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _label(text: str) -> dict:
    lowered = text.lower()
    category = next((c for c, words in _KEYWORDS if any(w in lowered for w in words)), "other")
    priority = "high" if any(w in lowered for w in _URGENT) else ("medium" if category != "other" else "low")
    first_line = text.strip().splitlines()[0][:120] if text.strip() else "empty ticket"
    return {
        "category": category,
        "priority": priority,
        "summary": f"Customer reports: {first_line.rstrip('.')}.",
    }

"""Runtime settings, read from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    database_path: str = "tickets.db"

    # How many classifications may run at once (= number of worker tasks).
    worker_concurrency: int = 4

    # Retry policy. `max_attempts` counts every claim, including ones lost to a crash.
    max_attempts: int = 3
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0

    # A single model call is abandoned after this long.
    llm_timeout_s: float = 10.0
    # A claimed ticket is reclaimable by another worker after this long.
    # Must comfortably exceed llm_timeout_s, otherwise a slow-but-alive call
    # could be picked up twice (fencing makes that safe, but wasteful).
    lease_s: float = 60.0

    # Idle workers wake up at least this often to look for due retries.
    poll_interval_s: float = 0.5
    # On shutdown, wait this long for in-flight calls before cancelling them.
    shutdown_grace_s: float = 10.0

    # Fake model behaviour.
    fake_llm_failure_rate: float = 0.2
    fake_llm_seed: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_path=os.environ.get("DATABASE_PATH", cls.database_path),
            worker_concurrency=_env_int("WORKER_CONCURRENCY", cls.worker_concurrency),
            max_attempts=_env_int("MAX_ATTEMPTS", cls.max_attempts),
            retry_base_delay_s=_env_float("RETRY_BASE_DELAY_S", cls.retry_base_delay_s),
            retry_max_delay_s=_env_float("RETRY_MAX_DELAY_S", cls.retry_max_delay_s),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", cls.llm_timeout_s),
            lease_s=_env_float("LEASE_S", cls.lease_s),
            poll_interval_s=_env_float("POLL_INTERVAL_S", cls.poll_interval_s),
            shutdown_grace_s=_env_float("SHUTDOWN_GRACE_S", cls.shutdown_grace_s),
            fake_llm_failure_rate=_env_float("FAKE_LLM_FAILURE_RATE", cls.fake_llm_failure_rate),
            fake_llm_seed=_env_int("FAKE_LLM_SEED", cls.fake_llm_seed),
        )

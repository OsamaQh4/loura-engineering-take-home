"""SQLite-backed ticket store.

The table doubles as the work queue. A worker *claims* a pending ticket by
writing a random lease token and an expiry time; every later write for that
attempt is conditional on still holding the token ("fencing"). This gives us:

* crash recovery for free: an expired lease is simply claimable again;
* no double-writes: a worker whose lease was taken over cannot overwrite the
  newer result, because its token no longer matches.

The schema also enforces the allowed category/priority values with CHECK
constraints, as a last line of defence behind the classifier's validation.
"""

from __future__ import annotations

import base64
import json
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from app.models import Category, Classification, Priority, Status, Ticket, TicketIn

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    id               TEXT    NOT NULL UNIQUE,
    subject          TEXT    NOT NULL,
    body             TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'classified', 'failed')),
    category         TEXT    CHECK (category IN ('billing', 'technical', 'account', 'other')),
    priority         TEXT    CHECK (priority IN ('low', 'medium', 'high')),
    summary          TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_error       TEXT,
    next_attempt_at  REAL    NOT NULL,
    lease_token      TEXT,
    lease_expires_at REAL,
    created_at       REAL    NOT NULL,
    updated_at       REAL    NOT NULL,
    -- a ticket is 'classified' if and only if it has a complete classification
    CHECK ((status = 'classified') =
           (category IS NOT NULL AND priority IS NOT NULL AND summary IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS ix_tickets_queue    ON tickets (status, next_attempt_at);
CREATE INDEX IF NOT EXISTS ix_tickets_category ON tickets (category, seq);
CREATE INDEX IF NOT EXISTS ix_tickets_priority ON tickets (priority, seq);
"""


class CreateResult(str, Enum):
    created = "created"
    exists = "exists"  # same id, same content: idempotent replay
    conflict = "conflict"  # same id, different content


@dataclass(frozen=True)
class Claim:
    ticket_id: str
    subject: str
    body: str
    attempts: int  # including this one
    lease_token: str


class InvalidCursor(ValueError):
    pass


class TicketStore:
    def __init__(self, path: str):
        self._path = path
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A connection per operation keeps this safe to call from any thread.
        # isolation_level=None: we issue BEGIN ourselves where it matters.
        conn = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- ingest

    def create(self, ticket: TicketIn, now: float) -> tuple[CreateResult, Ticket]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO tickets (id, subject, body, next_attempt_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO NOTHING
                """,
                (ticket.id, ticket.subject, ticket.body, now, now, now),
            )
            row = self._get_row(conn, ticket.id)
        assert row is not None
        if cur.rowcount == 1:
            return CreateResult.created, _to_ticket(row)
        if row["subject"] == ticket.subject and row["body"] == ticket.body:
            return CreateResult.exists, _to_ticket(row)
        return CreateResult.conflict, _to_ticket(row)

    # ------------------------------------------------------------------ read

    def get(self, ticket_id: str) -> Ticket | None:
        with self._connect() as conn:
            row = self._get_row(conn, ticket_id)
        return _to_ticket(row) if row else None

    def list(
        self,
        *,
        category: Category | None = None,
        priority: Priority | None = None,
        status: Status | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[Ticket], str | None]:
        """Newest first, keyset-paginated on `seq` so pages stay stable under inserts."""
        clauses, params = [], []
        if category:
            clauses.append("category = ?")
            params.append(category.value)
        if priority:
            clauses.append("priority = ?")
            params.append(priority.value)
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if cursor:
            clauses.append("seq < ?")
            params.append(_decode_cursor(cursor))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tickets {where} ORDER BY seq DESC LIMIT ?",
                (*params, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = _encode_cursor(rows[-1]["seq"]) if has_more else None
        return [_to_ticket(r) for r in rows], next_cursor

    # ----------------------------------------------------------- work queue

    def claim_next(self, now: float, lease_s: float) -> Claim | None:
        """Atomically take the next due pending ticket that nobody holds a live lease on."""
        token = secrets.token_hex(16)
        with self._connect() as conn:
            # IMMEDIATE takes the write lock up front, so the SELECT and the
            # UPDATE see the same state and two workers can't pick the same row.
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    UPDATE tickets
                       SET lease_token = ?, lease_expires_at = ?,
                           attempts = attempts + 1, updated_at = ?
                     WHERE seq = (
                           SELECT seq FROM tickets
                            WHERE status = 'pending'
                              AND next_attempt_at <= ?
                              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                            ORDER BY next_attempt_at, seq
                            LIMIT 1)
                    RETURNING id, subject, body, attempts
                    """,
                    (token, now + lease_s, now, now, now),
                ).fetchone()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        if row is None:
            return None
        return Claim(row["id"], row["subject"], row["body"], row["attempts"], token)

    def complete(self, claim: Claim, result: Classification, now: float) -> bool:
        return self._fenced_update(
            claim,
            """status = 'classified', category = ?, priority = ?, summary = ?,
               last_error = NULL""",
            (result.category.value, result.priority.value, result.summary),
            now,
        )

    def schedule_retry(self, claim: Claim, error: str, retry_at: float, now: float) -> bool:
        return self._fenced_update(
            claim, "last_error = ?, next_attempt_at = ?", (error, retry_at), now
        )

    def fail(self, claim: Claim, error: str, now: float) -> bool:
        return self._fenced_update(claim, "status = 'failed', last_error = ?", (error,), now)

    def release(self, claim: Claim, now: float) -> bool:
        """Hand an interrupted ticket straight back to the queue (graceful shutdown).

        The attempt is refunded: being interrupted is not the model's fault.
        """
        return self._fenced_update(claim, "attempts = attempts - 1", (), now)

    def _fenced_update(self, claim: Claim, set_sql: str, params: tuple, now: float) -> bool:
        """Apply an update only if `claim` still holds the lease. Always clears the lease."""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE tickets
                   SET {set_sql}, lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                 WHERE id = ? AND lease_token = ? AND status = 'pending'
                """,
                (*params, now, claim.ticket_id, claim.lease_token),
            )
        return cur.rowcount == 1

    @staticmethod
    def _get_row(conn: sqlite3.Connection, ticket_id: str) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()


def _to_ticket(row: sqlite3.Row) -> Ticket:
    classification = None
    if row["status"] == Status.classified.value:
        classification = Classification(
            category=row["category"], priority=row["priority"], summary=row["summary"]
        )
    return Ticket(
        id=row["id"],
        subject=row["subject"],
        body=row["body"],
        status=row["status"],
        classification=classification,
        attempts=row["attempts"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _encode_cursor(seq: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"s": seq}).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        seq = json.loads(base64.urlsafe_b64decode(cursor.encode()))["s"]
    except Exception as exc:
        raise InvalidCursor(cursor) from exc
    if not isinstance(seq, int):
        raise InvalidCursor(cursor)
    return seq

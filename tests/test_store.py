"""Storage-level guarantees: idempotent ingest, exclusive claims, fencing, DB-level constraints."""

import sqlite3

import pytest

from app.models import Category, Classification, Priority, Status
from app.store import CreateResult, InvalidCursor
from tests.conftest import ticket

CLS = Classification(category=Category.billing, priority=Priority.high, summary="s")


def test_create_is_idempotent(store):
    assert store.create(ticket(1), 0)[0] is CreateResult.created
    assert store.create(ticket(1), 1)[0] is CreateResult.exists
    assert store.create(ticket(1, body="different"), 2)[0] is CreateResult.conflict
    items, _ = store.list()
    assert len(items) == 1 and items[0].body == "Please refund."


def test_claim_is_exclusive_while_lease_is_live(store):
    store.create(ticket(1), 0)
    first = store.claim_next(now=10, lease_s=60)
    assert first is not None and first.attempts == 1
    assert store.claim_next(now=20, lease_s=60) is None


def test_expired_lease_is_reclaimable_and_stale_worker_is_fenced_out(store):
    """Simulates a worker that hangs/crashes: another takes over, the original can't overwrite."""
    store.create(ticket(1), 0)
    stale = store.claim_next(now=10, lease_s=60)
    fresh = store.claim_next(now=71, lease_s=60)
    assert fresh is not None and fresh.attempts == 2

    assert store.complete(stale, CLS, now=72) is False
    assert store.get("T-1").status is Status.pending
    assert store.complete(fresh, CLS, now=73) is True
    assert store.get("T-1").status is Status.classified


def test_terminal_tickets_are_never_claimed_again(store):
    store.create(ticket(1), 0)
    claim = store.claim_next(now=1, lease_s=60)
    store.complete(claim, CLS, now=2)
    assert store.claim_next(now=10_000, lease_s=60) is None


def test_retry_is_not_claimable_before_it_is_due(store):
    store.create(ticket(1), 0)
    claim = store.claim_next(now=1, lease_s=60)
    store.schedule_retry(claim, "boom", retry_at=50, now=2)
    assert store.claim_next(now=49, lease_s=60) is None
    assert store.claim_next(now=50, lease_s=60) is not None


def test_release_refunds_the_attempt(store):
    store.create(ticket(1), 0)
    claim = store.claim_next(now=1, lease_s=60)
    assert store.release(claim, now=2)
    again = store.claim_next(now=3, lease_s=60)
    assert again.attempts == 1


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE tickets SET status='classified', category='refunds', priority='high', summary='s'",
        "UPDATE tickets SET status='classified', category='billing', priority='urgent', summary='s'",
        "UPDATE tickets SET status='classified'",  # classified without a classification
        "UPDATE tickets SET status='done'",
    ],
)
def test_database_rejects_invalid_values_even_if_code_has_a_bug(store, settings, sql):
    store.create(ticket(1), 0)
    conn = sqlite3.connect(settings.database_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)
    conn.close()


def test_list_filters_and_paginates_without_gaps_or_duplicates(store):
    for i in range(25):
        store.create(ticket(i), i)
        claim = store.claim_next(now=i, lease_s=60)
        cat = Category.billing if i % 2 else Category.technical
        store.complete(claim, Classification(category=cat, priority=Priority.low, summary="s"), now=i)

    seen, cursor = [], None
    while True:
        items, cursor = store.list(category=Category.billing, limit=5, cursor=cursor)
        seen += [t.id for t in items]
        if cursor is None:
            break
    assert seen == [f"T-{i}" for i in range(24, -1, -1) if i % 2]  # newest first, all 12, once each
    assert store.list(category=Category.billing, priority=Priority.high)[0] == []


def test_bad_cursor_is_rejected(store):
    with pytest.raises(InvalidCursor):
        store.list(cursor="not-a-cursor")

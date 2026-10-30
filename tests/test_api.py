"""HTTP contract: status codes, idempotency, async classification, error format, listing."""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import GOOD, ScriptedLLM


@pytest.fixture
def llm():
    return ScriptedLLM(GOOD)


@pytest.fixture
def client(settings, llm):
    with TestClient(create_app(settings, llm)) as c:
        yield c


def post(client, id="T-1", subject="Charged twice", body="Please refund."):
    return client.post("/tickets", json={"id": id, "subject": subject, "body": body})


def wait_classified(client, id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = client.get(f"/tickets/{id}").json()
        if t["status"] != "pending":
            return t
        time.sleep(0.02)
    raise AssertionError(f"{id} still pending")


def test_create_returns_pending_and_classifies_in_background(client):
    r = post(client)
    assert r.status_code == 202
    assert r.json()["status"] == "pending" and r.json()["classification"] is None
    t = wait_classified(client, "T-1")
    assert t["status"] == "classified"
    assert t["classification"] == {"category": "billing", "priority": "high", "summary": "Customer was charged twice."}


def test_resubmitting_same_ticket_does_not_duplicate_or_reclassify(client, llm):
    assert post(client).status_code == 202
    wait_classified(client, "T-1")
    r = post(client)
    assert r.status_code == 200 and r.json()["status"] == "classified"
    time.sleep(0.1)
    assert llm.calls == 1
    assert len(client.get("/tickets").json()["items"]) == 1


def test_reusing_an_id_with_different_content_is_a_conflict(client):
    post(client)
    r = post(client, body="something else")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "id_conflict"


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "s", "body": "b"},
        {"id": "", "subject": "s", "body": "b"},
        {"id": "has spaces", "subject": "s", "body": "b"},
        {"id": "T-1", "subject": "s", "body": ""},
        {"id": "T-1", "subject": "s", "body": "b" * 50_001},
    ],
)
def test_invalid_ticket_is_rejected_with_error_envelope(client, payload):
    r = client.post("/tickets", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert r.json()["error"]["details"]


def test_unknown_ticket_is_404(client):
    r = client.get("/tickets/nope")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found", "message": "ticket 'nope' not found"}}


def test_list_filters_and_pages(client):
    for i in range(5):
        post(client, id=f"T-{i}")
    for i in range(5):
        wait_classified(client, f"T-{i}")

    page1 = client.get("/tickets", params={"category": "billing", "priority": "high", "limit": 3}).json()
    page2 = client.get("/tickets", params={"category": "billing", "limit": 3, "cursor": page1["next_cursor"]}).json()
    ids = [t["id"] for t in page1["items"] + page2["items"]]
    assert ids == ["T-4", "T-3", "T-2", "T-1", "T-0"]
    assert page2["next_cursor"] is None
    assert client.get("/tickets", params={"category": "account"}).json()["items"] == []


@pytest.mark.parametrize(
    "params, code",
    [
        ({"category": "refunds"}, "validation_error"),
        ({"priority": "urgent"}, "validation_error"),
        ({"limit": 0}, "validation_error"),
        ({"cursor": "garbage"}, "invalid_cursor"),
    ],
)
def test_bad_list_parameters(client, params, code):
    r = client.get("/tickets", params=params)
    assert r.status_code in (400, 422)
    assert r.json()["error"]["code"] == code

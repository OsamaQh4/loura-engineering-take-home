# Ticket classifier

This is a small HTTP service. It ingests support tickets, classifies each one in the background with an LLM (category, priority and a one-sentence summary), and serves the results back.

The service treats the model as an unreliable dependency. Every response is parsed and validated before anything is stored. Bad responses are retried, and a ticket that keeps getting bad responses ends up `failed` with a reason. The database itself also refuses out-of-range values.

## Run it

Requires Python 3.11+.

```bash
pip install -r requirements.txt
uvicorn app.main:create_app --factory            # http://localhost:8000, docs at /docs
python scripts/load_samples.py                   # in another terminal
python -m pytest
```

There are no API keys. The service uses a fake model (`app/llm.py`) that labels tickets with keyword rules. About 20% of the time it misbehaves the way real models do:
- provider errors
- hangs
- prose instead of JSON
- out-of-range labels
- missing fields
- arrays instead of objects

It also sometimes "falls for" prompt injection. You can tune it with `FAKE_LLM_FAILURE_RATE` and `FAKE_LLM_SEED`. All settings are in `app/config.py`.

## API

| Method & path | Result |
|---|---|
| `POST /tickets` `{id, subject, body}` | `202` new ticket (status `pending`), `200` same id and same content (idempotent replay, nothing re-run), `409` same id with *different* content, `422` invalid input |
| `GET /tickets/{id}` | `200` or `404` |
| `GET /tickets?category=&priority=&status=&limit=&cursor=` | `{items, next_cursor}`, newest first. `limit` ≤ 200. Pass `next_cursor` back to get the next page. |

Every error has the same shape: `{"error": {"code", "message", "details?"}}`.

A ticket's `status` is `pending`, then either `classified` (with `classification`) or `failed` (with `last_error`). `attempts` counts the model calls made.

## Layout

```
app/models.py      domain types and enums
app/store.py       SQLite store; the tickets table is also the work queue
app/llm.py         model interface + fake model
app/classifier.py  prompt building and output validation (the model boundary)
app/worker.py      worker pool: claim → call → validate → store / retry / fail
app/main.py        HTTP routes and wiring
```

## Decisions

**Storage: SQLite (one file, WAL mode).** It needs no setup and survives restarts, and transactions plus `UPDATE … RETURNING` are enough to build a correct queue. The schema has CHECK constraints on `status`, `category` and `priority`, plus a rule that a ticket is `classified` *if and only if* all three classification fields are present. So even a bug in my validation code cannot write `"refunds"` into the table. The cost is one host and one writer at a time (see Weaknesses).

**Idempotency.** `id` is `UNIQUE`, and inserts use `ON CONFLICT DO NOTHING`, so duplicates are impossible even under concurrent requests. Re-submitting the same content returns the existing ticket with `200` and never re-runs classification. Re-using an id with *different* content returns `409`. Silently keeping the old content would hide a bug in the caller.

**Async execution: the database table is the queue; the workers are asyncio tasks in the API process.**
- `POST` inserts a row and wakes the workers. The model is never called inside the request.
- A worker *claims* a ticket in a single `BEGIN IMMEDIATE` transaction. The claim writes a random lease token and a lease expiry, and increments `attempts`.
- Every later write for that attempt only applies `WHERE lease_token = <mine>` (fencing).

I chose this over Celery, Redis or a separate queue because it needs no extra infrastructure. The queue and the data also can't disagree, since the queue is the data.

**Concurrency: a fixed pool of `WORKER_CONCURRENCY` workers (default 4).** The real limit would be the provider's rate limit, so the number is configurable rather than clever. A test checks that peak concurrency never exceeds the pool size.

**Restarts and in-flight work.** Nothing is held in memory: a ticket in flight is just a `pending` row with a lease.
- **Clean shutdown** (the stretch item I picked): workers stop claiming and in-flight calls get `SHUTDOWN_GRACE_S` to finish. Calls still running after that are cancelled, and each hands its ticket straight back to the queue with its attempt refunded.
- **Crash or `kill -9`:** the lease expires after `LEASE_S` (60s) and any worker reclaims the ticket.
- **No double-processing:** if a slow "dead" worker comes back after its ticket was reclaimed, its write is fenced out and discarded. The worst case is one wasted model call. A result can never be stored twice or overwrite a newer one.
- **Crash loops:** a ticket that crashes the process every time still counts attempts, because they are incremented at claim time. It goes to `failed` instead of looping forever.

**Retry policy.** Each call has a timeout (`LLM_TIMEOUT_S`, 10s). These all count as a failed attempt:
- a timeout
- a provider error
- output that fails validation

A failed attempt is retried with exponential backoff and jitter (1s, 2s, … capped at 30s) up to `MAX_ATTEMPTS` (3). After that the ticket is `failed` and `last_error` says why, for example `invalid model output: invalid or missing field(s): priority`. I retry invalid output because sampling is non-deterministic, so the same prompt often works the second time. Raw invalid output is logged (truncated) for debugging, not stored.

**Validation (`app/classifier.py`).**
- **Accepted:** one JSON object, optionally wrapped in a ```` ``` ```` fence. I normalise case and whitespace on labels (`" Billing "` becomes `billing`), collapse whitespace in the summary, and cap it at 300 characters. Extra keys are ignored.
- **Rejected:** synonyms (`refunds`), out-of-range values, missing or null fields, non-objects, and JSON surrounded by chatter. I chose to tolerate formatting, not meaning.
- **Not enforced:** "one sentence" is not checked beyond the length cap; sentence-splitting heuristics were more fragile than useful.

**Prompt injection.** Validation is what limits the damage. The model can only change three fields, each restricted to a fixed set of values or bounded text. Whatever a ticket says, the worst outcome is a valid-but-wrong label or a misleading summary. It can never be a new status, a different ticket, an action, or a value outside the schema. On the prompt side:
- the ticket is JSON-encoded into the user message, so the text can't close a delimiter and start a new section;
- the system prompt says the ticket is data, not instructions.

`T-1007` is the injection sample; a test runs all samples through a gullible fake model and checks that everything ends in a valid state. I did **not** add keyword-based injection detection, because it is easy to bypass and produces false positives on real tickets.

## Weaknesses (honest list)

- **Injection that stays inside the schema still works.** A ticket can talk the model into `priority: high`, and validation can't tell. With the fake model, `T-1007` comes out as `billing` because it mentions a refund. Anything acting on priority automatically (paging someone, for example) should not trust it blindly.
- **The summary is model text derived from public input.** It's bounded and stripped of control characters, but downstream systems must still escape it before rendering it as HTML.
- **SQLite means one host.** Workers live in the API process, so scaling past one machine means moving to Postgres (where the same claim query works with `FOR UPDATE SKIP LOCKED`) and probably a separate worker process.
- **Leases use wall-clock time.** A big clock jump could expire leases early. Fencing keeps that safe, but it would cause wasted calls.
- **`pending` covers both "queued" and "being processed".** I kept to the three statuses in the spec, so callers can't see that a ticket is in flight or how long it has waited.
- **Failed is final.** There is no endpoint to re-queue failed tickets or re-classify after a prompt change. I picked graceful shutdown as the stretch item instead.
- **No `LEASE_S` > `LLM_TIMEOUT_S` check at startup**, no authentication, no rate limiting on ingest, and no request size limit beyond field lengths.
- **Retries use the identical prompt.** A deterministic model at temperature 0 could fail the same way three times.
- **The fake model's labels are keyword rules,** so they say nothing about classification quality.

## With more time

1. A re-classify endpoint and a `prompt_version` column, so tickets can be re-run after a prompt change. (The constant already exists in `classifier.py`.)
2. A small labelled evaluation set and a script that reports agreement per category and per priority.
3. A real provider client behind the same `LLMClient` interface, using structured output or tool calling. This reduces malformed output, though it doesn't remove the need for validation.
4. A "repair" retry that tells the model which field was invalid.
5. Metrics: queue depth, age of the oldest pending ticket, failure reasons, model latency.
6. Postgres and a separate worker process once one host is no longer enough.

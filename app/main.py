"""HTTP API and application wiring.

    POST /tickets           ingest (202 new, 200 idempotent replay, 409 id reused with different content)
    GET  /tickets/{id}      fetch one
    GET  /tickets           list; filters: category, priority, status; paging: limit, cursor
    GET  /healthz

Errors always look like: {"error": {"code": "...", "message": "...", "details": ...}}
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings
from app.llm import FakeLLM, LLMClient
from app.models import Category, Priority, Status, Ticket, TicketIn, TicketPage
from app.store import CreateResult, InvalidCursor, TicketStore
from app.worker import WorkerPool


class ApiError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.code = code


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = TicketStore(settings.database_path)
    workers = WorkerPool(
        store,
        llm or FakeLLM(failure_rate=settings.fake_llm_failure_rate, seed=settings.fake_llm_seed),
        settings,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        workers.start()  # also picks up tickets left pending by a previous run
        yield
        await workers.stop()

    app = FastAPI(title="Loura ticket classifier", lifespan=lifespan)
    app.state.store = store
    app.state.workers = workers

    @app.post("/tickets", response_model=Ticket, status_code=202)
    def create_ticket(body: TicketIn) -> JSONResponse:
        result, ticket = store.create(body, time.time())
        if result is CreateResult.conflict:
            raise ApiError(409, "id_conflict", f"ticket {body.id!r} already exists with different content")
        if result is CreateResult.created:
            workers.notify()
        status = 202 if result is CreateResult.created else 200
        return JSONResponse(ticket.model_dump(mode="json"), status_code=status)

    @app.get("/tickets/{ticket_id}", response_model=Ticket)
    def get_ticket(ticket_id: str) -> Ticket:
        ticket = store.get(ticket_id)
        if ticket is None:
            raise ApiError(404, "not_found", f"ticket {ticket_id!r} not found")
        return ticket

    @app.get("/tickets", response_model=TicketPage)
    def list_tickets(
        category: Category | None = None,
        priority: Priority | None = None,
        status: Status | None = None,
        limit: int = Query(50, ge=1, le=200),
        cursor: str | None = None,
    ) -> TicketPage:
        try:
            items, next_cursor = store.list(
                category=category, priority=priority, status=status, limit=limit, cursor=cursor
            )
        except InvalidCursor:
            raise ApiError(400, "invalid_cursor", "cursor is not valid") from None
        return TicketPage(items=items, next_cursor=next_cursor)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = getattr(exc, "code", "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return _error(422, "validation_error", "request is invalid", details)

    return app


def _error(status: int, code: str, message: str, details: object = None) -> JSONResponse:
    error: dict = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return JSONResponse({"error": error}, status_code=status)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Run with: uvicorn app.main:create_app --factory

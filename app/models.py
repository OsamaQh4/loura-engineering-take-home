"""Domain types shared by the API, the classifier and the store."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    pending = "pending"
    classified = "classified"
    failed = "failed"


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Classification(BaseModel):
    """A classification that has passed validation. Only this type is ever stored."""

    category: Category
    priority: Priority
    summary: str


class TicketIn(BaseModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    subject: str = Field(min_length=1, max_length=1_000)
    body: str = Field(min_length=1, max_length=50_000)


class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    status: Status
    classification: Classification | None
    attempts: int
    last_error: str | None
    created_at: float
    updated_at: float


class TicketPage(BaseModel):
    items: list[Ticket]
    next_cursor: str | None

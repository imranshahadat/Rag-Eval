"""Request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique ID for each user conversation")
    message: str = Field(..., min_length=1, description="Message sent by the user")

    @field_validator("session_id", "message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty or whitespace")
        return v


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_response: str


class ContactResponse(BaseModel):
    session_id: str
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    updated_at: str | None = None


class SessionClearResponse(BaseModel):
    session_id: str
    cleared: bool


class SessionStartRequest(BaseModel):
    session_id: str | None = None
    name: str | None = None
    email: str | None = None
    phone: str | None = None


class SessionStartResponse(BaseModel):
    session_id: str
    recognized: bool
    name: str | None = None
    email: str | None = None
    past_turns: list[dict] = Field(default_factory=list)

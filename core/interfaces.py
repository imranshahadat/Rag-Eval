"""Shared abstractions and data transfer objects.

Consumer code depends on these protocols (Dependency Inversion Principle).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Turn:
    """A single user/assistant exchange."""

    user_query: str
    assistant_response: str
    session_id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class ContactInfo:
    """Contact details a user volunteered during chat."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None

    def has_any(self) -> bool:
        return any((self.name, self.email, self.phone))


@dataclass(frozen=True)
class UserProfile:
    """A known user plus their long-term memory."""

    user_id: str
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    long_term_summary: str = ""
    created_at: str | None = None
    updated_at: str | None = None


class ChatRepository(Protocol):
    """Persistence operations required by the application (Interface Segregation)."""

    # chat history
    def save_turn(self, session_id: str, user_query: str, assistant_response: str,
                  user_id: str | None = None) -> bool: ...
    def get_history(self, session_id: str, limit: int = 50) -> list[Turn]: ...
    def get_user_turns(self, user_id: str, limit: int = 12) -> list[Turn]: ...
    def get_user_history(self, user_id: str, limit: int = 6,
                         exclude_session: str | None = None) -> list[tuple[str, str]]: ...

    # contacts (per session)
    def upsert_contact(self, session_id: str, contact: ContactInfo) -> bool: ...
    def get_contact(self, session_id: str) -> ContactInfo | None: ...
    def get_all_contacts(self) -> list[ContactInfo]: ...

    # identity / long-term memory
    def find_user_by_contact(self, email: str | None = None, phone: str | None = None) -> str | None: ...
    def upsert_profile(self, user_id: str, contact: ContactInfo) -> bool: ...
    def get_profile(self, user_id: str) -> UserProfile | None: ...
    def get_all_profiles(self) -> list[UserProfile]: ...
    def bind_session(self, session_id: str, user_id: str) -> bool: ...
    def get_session_user(self, session_id: str) -> str | None: ...
    def backfill_session_user(self, session_id: str, user_id: str) -> None: ...
    def update_long_term_summary(self, user_id: str, summary: str) -> bool: ...

    def close(self) -> None: ...

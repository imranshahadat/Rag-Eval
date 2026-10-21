"""Identity resolution and long-term memory maintenance.

Resolves WHO the user is across sessions:
  1. Extract contact info volunteered in the message (ContactExtractor).
  2. user_id = contact from message > session binding > None (anonymous).
  3. First time this user_id appears in this session -> replay past-session turns.
  4. Every few turns -> merge session context into the long-term summary.

Anonymous users get None: chat still works, persistence stays per-session.
"""
from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

from core.interfaces import ChatRepository, ContactInfo, UserProfile
from memory.contact_extractor import ContactExtractor
from memory.short_term import ChatMemoryManager

logger = logging.getLogger(__name__)


def make_user_id(email: str | None, phone: str | None) -> str | None:
    """Identity key: email (fallback: phone)."""
    if email:
        return email.strip().lower()
    if phone:
        return f"phone:{phone}"
    return None


class IdentityManager:
    SUMMARY_EVERY_N_MESSAGES = 6  # update long-term summary every ~3 turns

    ASK_CONTACT_RULE = (
        "IDENTITY RULES: You don't know this user yet. Since this is the first reply of the "
        "conversation, greet them briefly and ask for their name, email and phone number so you "
        "can recognize and remember them in future conversations. Make clear it is optional, "
        "then invite them to ask anything. If they skip it or change the subject, just keep "
        "helping - never block the chat and never ask again in later replies.\n\n"
    )

    MERGE_PROMPT = (
        "Merge the NEW conversation excerpt into the user's LONG-TERM MEMORY. "
        "Keep durable facts only: their name/contact details, who they are, topics they asked "
        "about, preferences, goals, decisions, and anything worth recalling next time they chat. "
        "Drop small talk and filler. Be concise (max ~150 words). "
        "Output ONLY the merged memory text.\n\n"
        "LONG-TERM MEMORY:\n{old}\n\n"
        "NEW CONVERSATION EXCERPT:\n{new}"
    )

    def __init__(self, db: ChatRepository, llm: BaseChatModel, extractor: ContactExtractor):
        self._db = db
        self._llm = llm
        self._extractor = extractor
        self._hydrated: dict[str, str] = {}          # session_id -> user_id already loaded
        self._last_summary_len: dict[str, int] = {}  # session_id -> history len at last summary
        self._pending_names: dict[str, str] = {}     # name-only messages before email arrives

    # ---------- per-message resolution (call BEFORE answering) ----------

    def resolve(self, session_id: str, message: str,
                memory: ChatMemoryManager) -> UserProfile | None:
        contact = self._extractor.extract(message)

        # name-only message (no email/phone yet) -> keep until identity arrives
        if contact and contact.name and not make_user_id(contact.email, contact.phone):
            self._pending_names.setdefault(session_id, contact.name)

        user_id = make_user_id(contact.email, contact.phone) if contact else None
        if not user_id:
            user_id = self._db.get_session_user(session_id)

        if not user_id:
            return None  # anonymous - chat continues, persistence stays per-session

        if contact:
            self._db.upsert_profile(user_id, contact)
        pending = self._pending_names.pop(session_id, None)
        if pending:
            self._db.upsert_profile(user_id, ContactInfo(name=pending))

        self._db.bind_session(session_id, user_id)
        self._db.backfill_session_user(session_id, user_id)

        # hydrate once per session (or if the user switches identity mid-session)
        if self._hydrated.get(session_id) != user_id:
            past = self._db.get_user_history(user_id, limit=6, exclude_session=session_id)
            memory.preload(past)
            self._hydrated[session_id] = user_id
            self._last_summary_len[session_id] = len(memory.chat_history)

        return self._db.get_profile(user_id)

    # ---------- prompt context ----------

    def build_context(self, profile: UserProfile | None, is_first_turn: bool) -> str:
        if not profile:
            return self.ASK_CONTACT_RULE if is_first_turn else ""  # never nag anonymous users
        parts = [
            f"USER PROFILE: name={profile.name or 'unknown'}, "
            f"email={profile.email or 'unknown'}, phone={profile.phone or 'unknown'}."
        ]
        if profile.long_term_summary:
            parts.append(
                "LONG-TERM MEMORY (from this user's previous conversations):\n"
                + profile.long_term_summary
            )
        if profile.name:
            parts.append(
                "This is a returning user - greet them by name naturally and use the "
                "long-term memory to recall earlier topics. Never ask for their "
                "contact details again."
            )
        return "\n".join(parts) + "\n\n"

    # ---------- long-term summary maintenance ----------

    def maybe_update_summary(self, session_id: str, memory: ChatMemoryManager) -> None:
        user_id = self._db.get_session_user(session_id)
        if not user_id:
            return
        n = len(memory.chat_history)
        if n - self._last_summary_len.get(session_id, 0) < self.SUMMARY_EVERY_N_MESSAGES:
            return
        self._last_summary_len[session_id] = n
        self._merge_summary(user_id, memory.chat_history[-self.SUMMARY_EVERY_N_MESSAGES:])

    def flush_summary(self, session_id: str, memory: ChatMemoryManager) -> None:
        """Force-update at session end (DELETE /session)."""
        user_id = self._db.get_session_user(session_id)
        if not user_id:
            return
        since = self._last_summary_len.get(session_id, 0)
        recent = memory.chat_history[since:]
        if recent:
            self._merge_summary(user_id, recent)
        self._last_summary_len[session_id] = len(memory.chat_history)

    def _merge_summary(self, user_id: str, messages: list[BaseMessage]) -> None:
        try:
            profile = self._db.get_profile(user_id)
            old = profile.long_term_summary if profile else ""
            conversation = "\n".join(f"{m.type.upper()}: {m.content}" for m in messages)
            merged = self._llm.invoke(
                self.MERGE_PROMPT.format(old=old or "(empty)", new=conversation)
            ).content.strip()
            self._db.update_long_term_summary(user_id, merged)
            logger.info("Long-term memory updated for %s", user_id)
        except Exception as e:
            logger.warning("Long-term summary update skipped: %s", e)

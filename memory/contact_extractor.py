"""Extracts contact details a user volunteers in chat (Single Responsibility).

Nothing is extracted unless it appears in the user's own message.
"""
from __future__ import annotations

import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel

from core.interfaces import ContactInfo

logger = logging.getLogger(__name__)


class ContactExtractor:
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{7,}\d")

    # Strong signals - safe to trust directly
    STRONG_NAME_RE = re.compile(
        r"\b(?:my name is|call me)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})"
    )
    # Weak signals - verify with the LLM before storing
    WEAK_NAME_RE = re.compile(r"\b(?:i am|i'm|myself)\b", re.IGNORECASE)

    MAX_NAME_LEN = 50

    def __init__(self, llm: BaseChatModel | None):
        self._llm = llm

    def extract(self, message: str) -> ContactInfo | None:
        email = self._extract_email(message)
        phone = self._extract_phone(message)
        name = self._extract_name(message)
        contact = ContactInfo(name=name, email=email, phone=phone)
        return contact if contact.has_any() else None

    def _extract_email(self, message: str) -> str | None:
        match = self.EMAIL_RE.search(message)
        return match.group(0).lower() if match else None

    def _extract_phone(self, message: str) -> str | None:
        match = self.PHONE_RE.search(message)
        if not match:
            return None
        digits = re.sub(r"[^\d+]", "", match.group(0))
        # Basic sanity: 7-15 digits (E.164 allows up to 15)
        digit_count = len(digits.lstrip("+"))
        if 7 <= digit_count <= 15:
            return digits
        return None

    def _extract_name(self, message: str) -> str | None:
        strong = self.STRONG_NAME_RE.search(message)
        if strong:
            name = strong.group(1).strip(" .,!?")
            if 0 < len(name) <= self.MAX_NAME_LEN:
                return name.title()

        if self.WEAK_NAME_RE.search(message) and self._llm is not None:
            return self._extract_name_with_llm(message)

        return None

    def _extract_name_with_llm(self, message: str) -> str | None:
        """Fallback for ambiguous self-introductions. Never raises."""
        try:
            prompt = (
                "Does the user state their OWN name in the message below? "
                "If yes, reply with ONLY the name. If no, reply with exactly NONE.\n\n"
                f"Message: {message}"
            )
            answer = self._llm.invoke(prompt).content.strip().strip('"').strip(".")
            if not answer or answer.upper() == "NONE" or len(answer) > self.MAX_NAME_LEN:
                return None
            return answer
        except Exception as e:
            logger.warning("Name extraction skipped: %s", e)
            return None

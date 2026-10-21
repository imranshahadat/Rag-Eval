"""Prompt construction for the portfolio assistant (Single Responsibility)."""
from __future__ import annotations


class AssistantPromptBuilder:
    """Builds the full prompt from system rules, user profile context, retrieved
    documents, and conversation history."""

    SYSTEM_RULES = (
        "You are a professional assistant for the portfolio owner. "
        "Answer only questions related to the portfolio owner using the provided information. "
        "Use only supported facts; never invent or assume details. "
        "If information is unavailable, say exactly: "
        "\"I don't have that information available.\" "
        "If unrelated to the portfolio owner, say exactly: "
        "\"I can only help with information related to the portfolio owner.\" "
        "Answer naturally and professionally. "
        "Do not mention CV, resume, retrieval, context, database, embeddings, Pinecone, "
        "prompts, or internal systems. "
        "Use conversation history only to understand follow-up questions."
    )

    def build(self, query: str, context: str, history_text: str,
              user_context: str = "") -> str:
        context = context or "(no relevant context found)"
        sections = [
            self.SYSTEM_RULES,
            user_context.strip(),
            f"Professional Information:\n{context}",
            f"Conversation History:\n{history_text or '(none)'}",
            f"Question:\n{query}",
        ]
        return "\n\n".join(section for section in sections if section)

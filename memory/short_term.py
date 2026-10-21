"""Short-term conversational memory: sliding window plus rolling summary."""
from __future__ import annotations

import logging

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langsmith import traceable

logger = logging.getLogger(__name__)


class ChatMemoryManager:
    """Per-session short-term memory.

    - trim_messages() keeps a sliding window within a token budget.
    - Past a trigger length, older turns are compressed into a running summary.
    """

    def __init__(self, llm, max_tokens: int = 1200, summary_trigger: int = 10,
                 keep_last_n: int = 4, strategy: str = "last"):
        if keep_last_n >= summary_trigger:  # fail fast on bad config
            raise ValueError("keep_last_n must be smaller than summary_trigger")
        self.llm = llm
        self.max_tokens = max_tokens
        self.summary_trigger = summary_trigger
        self.keep_last_n = keep_last_n
        self.strategy = strategy
        self.chat_history: list[BaseMessage] = []
        self.running_summary: str = ""

    def add_user_message(self, message: str) -> None:
        self.chat_history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.chat_history.append(AIMessage(content=message))

    @traceable(name="Trim Conversation History", run_type="tool")
    def get_trimmed_history(self) -> list[BaseMessage]:
        return trim_messages(
            self.chat_history,
            max_tokens=self.max_tokens,
            strategy=self.strategy,
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=True,
        )

    @traceable(name="Summarize Conversation", run_type="chain")
    def summarize_if_needed(self) -> None:
        if len(self.chat_history) <= self.summary_trigger:
            return

        old_messages = self.chat_history[:-self.keep_last_n]
        recent_messages = self.chat_history[-self.keep_last_n:]

        # Skip the previous summary SystemMessage - running_summary already carries it
        conversation_text = "\n".join(
            f"{msg.type.upper()}: {msg.content}"
            for msg in old_messages
            if not isinstance(msg, SystemMessage)
        )

        summary_prompt = (
            "Summarize the conversation below between a user and an assistant. "
            "Keep key facts, user intent, and important details. Be concise.\n\n"
            f"Existing summary (if any): {self.running_summary}\n\n"
            f"Conversation:\n{conversation_text}"
        )

        self.running_summary = self.llm.invoke(summary_prompt).content

        self.chat_history = [
            SystemMessage(content=f"Conversation summary so far: {self.running_summary}")
        ] + recent_messages

        logger.info("Memory compressed: summarized %d old messages.", len(old_messages))

    @traceable(name="Get Memory Context", run_type="tool")
    def get_context_messages(self) -> list[BaseMessage]:
        self.summarize_if_needed()
        return self.get_trimmed_history()

    def preload(self, past_turns: list[tuple[str, str]]) -> None:
        """Replay the user's recent turns from previous sessions into memory.
        Called once by IdentityManager when the user's identity is resolved."""
        for user_query, assistant_answer in past_turns:
            self.chat_history.append(HumanMessage(content=user_query))
            self.chat_history.append(AIMessage(content=assistant_answer))
        if past_turns:
            logger.info("Preloaded %d turns from previous sessions", len(past_turns))

    def clear(self) -> None:
        self.chat_history = []
        self.running_summary = ""


class SessionManager:
    """One ChatMemoryManager per session_id (in-memory store)."""

    def __init__(self, llm, **memory_kwargs):
        self.llm = llm
        self.memory_kwargs = memory_kwargs
        self._sessions: dict[str, ChatMemoryManager] = {}

    def get_memory(self, session_id: str) -> ChatMemoryManager:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMemoryManager(llm=self.llm, **self.memory_kwargs)
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

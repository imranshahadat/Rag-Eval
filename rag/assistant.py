"""RAG assistant that answers questions about the portfolio owner."""
from __future__ import annotations

import logging
from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langsmith import traceable

from memory.short_term import ChatMemoryManager
from rag.prompt_builder import AssistantPromptBuilder

logger = logging.getLogger(__name__)


class PortfolioAssistant:
    """Retrieval-augmented assistant.

    One instance serves all sessions; short-term memory is passed in per call.
    Depends only on LangChain abstractions (Dependency Inversion Principle).
    """

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever,
                 prompt_builder: AssistantPromptBuilder | None = None):
        self._llm = llm
        self._retriever = retriever
        self._prompts = prompt_builder or AssistantPromptBuilder()

    @traceable(name="Assistant Respond", run_type="chain")
    def respond(self, query: str, memory: ChatMemoryManager, user_context: str = "") -> str:
        memory.add_user_message(query)
        answer = self._llm.invoke(self._build(query, memory, user_context)).content
        memory.add_ai_message(answer)
        return answer

    def stream(self, query: str, memory: ChatMemoryManager,
               user_context: str = "") -> Iterator[str]:
        memory.add_user_message(query)
        prompt = self._build(query, memory, user_context)
        full = ""
        for chunk in self._model_stream(prompt):
            full += chunk
            yield chunk
        memory.add_ai_message(full)

    def _model_stream(self, prompt: str) -> Iterator[str]:
        for chunk in self._llm.stream(prompt):
            yield chunk.content

    def _build(self, query: str, memory: ChatMemoryManager, user_context: str) -> str:
        history = memory.get_context_messages()
        history_text = "\n".join(f"{m.type.upper()}: {m.content}" for m in history)
        docs = self._retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        return self._prompts.build(query, context, history_text, user_context)

"""Chat model provider backed by Novita's OpenAI-compatible API."""
from __future__ import annotations

from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import Settings


class NovitaChatModel:
    """Builds and exposes a standard LangChain chat model.

    Consumers receive ``model`` (a BaseChatModel) so they never depend on
    this concrete wrapper (Dependency Inversion Principle).
    """

    def __init__(self, settings: Settings):
        if not settings.novita_api_key:
            raise ValueError("NOVITA_API_KEY is missing.")
        self._model: BaseChatModel = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.novita_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @property
    def model(self) -> BaseChatModel:
        return self._model

    def invoke(self, text: str) -> str:
        """One-shot Q&A convenience method."""
        return self._model.invoke(text).content

    def stream(self, text: str) -> Iterator[str]:
        """Yield response chunks one at a time."""
        for chunk in self._model.stream(text):
            yield chunk.content

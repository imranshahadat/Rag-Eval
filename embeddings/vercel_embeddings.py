"""Embeddings provider backed by the Vercel AI Gateway."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import Settings


class VercelEmbeddings(Embeddings):
    """LangChain-compatible embeddings client for the Vercel AI Gateway."""

    def __init__(self, settings: Settings):
        if not settings.ai_gateway_api_key:
            raise ValueError("AI_GATEWAY_API_KEY is missing.")
        self.model_name = settings.embedding_model
        self._client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.ai_gateway_api_key,
            base_url=settings.embedding_base_url,
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    @property
    def dimension(self) -> int:
        """Embedding vector size, measured from a live query."""
        return len(self.embed_query("dimension check"))

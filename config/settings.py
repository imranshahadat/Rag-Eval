"""Application configuration.

Settings are loaded from environment variables / .env file, with a fallback
to Google Colab userdata secrets, so the same codebase runs in VS Code and Colab.
"""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def read_secret(name: str) -> str | None:
    """Read a secret from the environment, falling back to Colab userdata."""
    value = os.getenv(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:
        return None


class Settings(BaseSettings):
    """Strongly typed application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LangSmith tracing
    langchain_api_key: str | None = None
    langsmith_project: str = "portfolio-assistant"

    # Embeddings (Vercel AI Gateway)
    embedding_model: str = "cohere/embed-v5.0-fast"
    embedding_base_url: str = "https://ai-gateway.vercel.sh/v1"
    ai_gateway_api_key: str | None = None

    # Chat model (Novita)
    llm_model: str = "apodex/apodex-1.1-mini"
    llm_base_url: str = "https://api.novita.ai/openai"
    novita_api_key: str | None = None
    llm_temperature: float = 0.8
    llm_max_tokens: int = 3000

    # Pinecone
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "rag-index"
    pinecone_namespace: str = "pdf-documents"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    recreate_index_on_dimension_mismatch: bool = False

    # Chunking / retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 3

    # Short-term memory
    memory_max_tokens: int = 1200
    memory_summary_trigger: int = 10
    memory_keep_last: int = 4

    # Persistence
    database_url: str | None = None

    # Misc
    log_dir: str = "logs"

    @classmethod
    def load(cls) -> "Settings":
        """Load settings, resolving secrets from env, .env, or Colab userdata."""
        settings = cls()
        settings.langchain_api_key = settings.langchain_api_key or read_secret("LANGCHAIN_API_KEY")
        settings.ai_gateway_api_key = (
            settings.ai_gateway_api_key or read_secret("AI_GATEWAY_API_KEY") or read_secret("VERCEL_API_KEY")
        )
        settings.novita_api_key = settings.novita_api_key or read_secret("NOVITA_API_KEY")
        settings.pinecone_api_key = settings.pinecone_api_key or read_secret("PINECONE_API_KEY")
        settings.database_url = settings.database_url or read_secret("DATABASE_URL")
        return settings

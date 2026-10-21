"""Pinecone index and vector store provider."""
from __future__ import annotations

import logging
import time

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config.settings import Settings

logger = logging.getLogger(__name__)


class PineconeVectorStoreProvider:
    """Owns the Pinecone client, index lifecycle, and the LangChain vector store."""

    def __init__(self, settings: Settings):
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is missing.")
        self._settings = settings
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._store: PineconeVectorStore | None = None

    def setup_index(self, dimension: int, metric: str = "cosine") -> None:
        """Create the index if missing; validate dimension against an existing index."""
        existing = self._pc.list_indexes().names()

        if self._settings.pinecone_index_name in existing:
            current_dim = self._pc.describe_index(self._settings.pinecone_index_name).dimension
            if current_dim == dimension:
                logger.info("Index '%s' exists, dimension matches (%d)",
                            self._settings.pinecone_index_name, dimension)
                return

            message = (f"Index '{self._settings.pinecone_index_name}' has dimension "
                       f"{current_dim}, but the model outputs {dimension}.")
            if self._settings.recreate_index_on_dimension_mismatch:
                logger.warning("%s Recreating index.", message)
                self.delete_index()
            else:
                # Fail fast here instead of a cryptic 400 at upsert time
                raise ValueError(
                    f"{message} Dimensions are immutable in Pinecone. "
                    f"Delete the index or set RECREATE_INDEX_ON_DIMENSION_MISMATCH=true."
                )

        self._pc.create_index(
            name=self._settings.pinecone_index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=self._settings.pinecone_cloud,
                                region=self._settings.pinecone_region),
        )
        while not self._pc.describe_index(self._settings.pinecone_index_name).status["ready"]:
            time.sleep(2)
        logger.info("Created index '%s' (dim=%d)", self._settings.pinecone_index_name, dimension)

    def get_vector_store(self, embedding_model: Embeddings) -> VectorStore:
        """Return a LangChain VectorStore bound to the configured index/namespace."""
        if self._store is None:
            self._store = PineconeVectorStore(
                index=self._pc.Index(self._settings.pinecone_index_name),
                embedding=embedding_model,
                namespace=self._settings.pinecone_namespace,
            )
        return self._store

    def delete_index(self) -> None:
        self._pc.delete_index(self._settings.pinecone_index_name)
        while self._settings.pinecone_index_name in self._pc.list_indexes().names():
            time.sleep(1)
        logger.info("Deleted index '%s'", self._settings.pinecone_index_name)

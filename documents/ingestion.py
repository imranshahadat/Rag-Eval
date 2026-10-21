"""End-to-end document ingestion pipeline: load, split, index."""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from documents.chunker import DocumentChunker
from documents.loaders import DocumentLoaderService

logger = logging.getLogger(__name__)


def ingest_documents(documents: list[Document], chunker: DocumentChunker,
                     vector_store: VectorStore) -> int:
    """Split documents and index them. Returns the number of chunks created."""
    return len(chunker.split_and_index(documents, vector_store))


def ingest_file(file_path: str, file_type: str, chunker: DocumentChunker,
                vector_store: VectorStore, loader: DocumentLoaderService | None = None) -> int:
    """Load a single file, split it, and index it. Returns the number of chunks created."""
    loader = loader or DocumentLoaderService()

    if file_type == "pdf":
        documents = loader.load_pdf(file_path)
    elif file_type == "text":
        documents = loader.load_text(file_path)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    if not documents:
        raise ValueError(f"No content extracted from {file_path}")

    return ingest_documents(documents, chunker, vector_store)

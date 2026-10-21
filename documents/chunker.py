"""Document chunking (Single Responsibility)."""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits documents into overlapping chunks using LangChain text splitters."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[Document], verbose: bool = True) -> list[Document]:
        if not documents:
            raise ValueError("No documents provided to split.")
        chunks = self._splitter.split_documents(documents)
        if verbose:
            self._log_stats(len(documents), chunks)
        return chunks

    def split_and_index(self, documents: list[Document],
                        vector_store: VectorStore) -> list[Document]:
        """Split documents and index the chunks into the vector store."""
        chunks = self.split(documents, verbose=True)
        vector_store.add_documents(documents=chunks)
        logger.info("Indexed %d chunks into the vector store", len(chunks))
        return chunks

    def _log_stats(self, doc_count: int, chunks: list[Document]) -> None:
        total_chars = sum(len(c.page_content) for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        logger.info(
            "Splitting stats | documents=%d | chunks=%d | chunk_size=%d | overlap=%d | "
            "total_chars=%d | avg_chunk=%.0f",
            doc_count, len(chunks), self.chunk_size, self.chunk_overlap, total_chars, avg_chars,
        )

"""Document loading from files and folders (Single Responsibility)."""
from __future__ import annotations

import logging

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoaderService:
    """Loads PDF / text documents from disk."""

    def load_pdf(self, file_path: str) -> list[Document]:
        documents = PyPDFLoader(file_path).load()
        logger.info("Loaded PDF from %s (%d pages)", file_path, len(documents))
        return documents

    def load_text(self, file_path: str) -> list[Document]:
        documents = TextLoader(file_path).load()
        logger.info("Loaded text file from %s", file_path)
        return documents

    def load_pdf_folder(self, folder_path: str, glob_pattern: str = "**/*.pdf") -> list[Document]:
        documents = DirectoryLoader(
            folder_path,
            glob=glob_pattern,
            loader_cls=PyPDFLoader,
            show_progress=True,
        ).load()
        logger.info("Loaded %d PDF documents from %s", len(documents), folder_path)
        return documents

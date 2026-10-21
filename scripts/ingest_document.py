"""CLI: ingest a PDF or text file into the vector store.

Usage (from the project root):
    python -m scripts.ingest_document --file data/resume.pdf --type pdf
    python -m scripts.ingest_document --file data/notes.txt --type text
"""
from __future__ import annotations

import argparse

from config.settings import Settings
from documents.chunker import DocumentChunker
from documents.ingestion import ingest_file
from embeddings.vercel_embeddings import VercelEmbeddings
from vectorstore.pinecone_store import PineconeVectorStoreProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into the vector store.")
    parser.add_argument("--file", required=True, help="Path to the file to ingest.")
    parser.add_argument("--type", choices=["pdf", "text"], default="pdf",
                        help="File type (default: pdf).")
    args = parser.parse_args()

    settings = Settings.load()
    embeddings = VercelEmbeddings(settings)
    provider = PineconeVectorStoreProvider(settings)
    provider.setup_index(dimension=embeddings.dimension)
    vector_store = provider.get_vector_store(embeddings)

    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    count = ingest_file(args.file, args.type, chunker, vector_store)
    print(f"Ingestion complete: {count} chunks indexed from {args.file}")


if __name__ == "__main__":
    main()

"""Composition root.

Wires concrete implementations (Novita, Vercel, Pinecone, Postgres) to their
consumers. Routes only ever touch this container or LangChain abstractions.
"""
from __future__ import annotations

from config.settings import Settings
from core.bootstrap import configure_tracing
from embeddings.vercel_embeddings import VercelEmbeddings
from llm.novita_chat import NovitaChatModel
from memory.contact_extractor import ContactExtractor
from memory.identity import IdentityManager
from memory.short_term import SessionManager
from persistence.postgres import PostgresDatabase
from rag.assistant import PortfolioAssistant
from vectorstore.pinecone_store import PineconeVectorStoreProvider


class AppContainer:
    """Holds every application component, fully wired."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        configure_tracing(self.settings)

        self.llm = NovitaChatModel(self.settings)
        self.embeddings = VercelEmbeddings(self.settings)

        self.vector_store_provider = PineconeVectorStoreProvider(self.settings)
        self.vector_store_provider.setup_index(dimension=self.embeddings.dimension)
        self.vector_store = self.vector_store_provider.get_vector_store(self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.settings.retriever_k}
        )

        self.database = PostgresDatabase(self.settings.database_url)

        self.contact_extractor = ContactExtractor(self.llm.model)
        self.identity = IdentityManager(self.database, self.llm.model, self.contact_extractor)
        self.session_manager = SessionManager(
            llm=self.llm.model,
            max_tokens=self.settings.memory_max_tokens,
            summary_trigger=self.settings.memory_summary_trigger,
            keep_last_n=self.settings.memory_keep_last,
        )
        self.assistant = PortfolioAssistant(llm=self.llm.model, retriever=self.retriever)

    def shutdown(self) -> None:
        self.database.close()

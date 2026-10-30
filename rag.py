"""
RAG (Retrieval-Augmented Generation) Service.
Manages vector embeddings, local Qdrant vector store connection,
and document retrieval with source attribution.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service encapsulating document retrieval from Qdrant vector store."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[QdrantVectorStore] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initialize embeddings with offline-first caching."""
        if self._embeddings is None:
            logger.info("Initializing embedding model: %s", self.settings.EMBEDDING_MODEL_NAME)
            try:
                # Fast path: use already downloaded cached model weights
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception:
                # Fallback to standard initialization if not yet in local cache
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
        return self._embeddings

    @property
    def vector_store(self) -> Optional[QdrantVectorStore]:
        """Connect to Qdrant vector store with error resilience."""
        if self._vector_store is None:
            try:
                logger.info(
                    "Connecting to Qdrant collection '%s' at '%s'",
                    self.settings.QDRANT_COLLECTION_NAME,
                    self.settings.QDRANT_PATH,
                )
                self._vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=self.embeddings,
                    collection_name=self.settings.QDRANT_COLLECTION_NAME,
                    path=self.settings.QDRANT_PATH,
                )
            except Exception as e:
                logger.warning(
                    "Qdrant vector store initialization note: %s. "
                    "(If running concurrently with another server, local lock is held by the active server).",
                    e,
                )
                self._vector_store = None
        return self._vector_store

    def get_retriever(self, top_k: Optional[int] = None):
        """Returns a configured retriever instance if vector store is ready."""
        k = top_k or self.settings.TOP_K_DOCUMENTS
        vs = self.vector_store
        if vs is not None:
            return vs.as_retriever(search_kwargs={"k": k})
        return None

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieves top-k relevant document chunks for the given query.
        Returns an empty list safely if retriever is unavailable.
        """
        retriever = self.get_retriever(top_k=top_k)
        if retriever is None:
            logger.warning("Retriever unavailable; returning empty context.")
            return []

        try:
            return retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving context for query '%s': %s", query, e)
            return []

    def health_check(self) -> Dict[str, Any]:
        """Checks the readiness of the vector store without forcing heavy file locks."""
        has_dir = os.path.exists(self.settings.QDRANT_PATH)
        is_connected = self._vector_store is not None
        return {
            "qdrant_path": self.settings.QDRANT_PATH,
            "collection_name": self.settings.QDRANT_COLLECTION_NAME,
            "storage_path_exists": has_dir,
            "connected": is_connected,
        }


# Global singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Returns the application RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

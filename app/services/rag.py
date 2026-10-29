"""
RAG (Retrieval-Augmented Generation) Service.
Manages vector embeddings, local Qdrant vector store connection,
document retrieval with source attribution, incremental indexing,
and document vector deletion/lifecycle.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service encapsulating document retrieval, vector indexing, and deletion with Qdrant."""

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
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception:
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

    def add_documents(self, documents: List[Document]) -> int:
        """
        Splits and incrementally adds documents into the active Qdrant collection.
        Returns the count of new chunks added.
        """
        if not documents:
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            return 0

        vs = self.vector_store
        if vs is None:
            logger.info("Initializing new Qdrant collection '%s' with %d chunks", self.settings.QDRANT_COLLECTION_NAME, len(chunks))
            self._vector_store = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                path=self.settings.QDRANT_PATH,
                collection_name=self.settings.QDRANT_COLLECTION_NAME,
            )
        else:
            vs.add_documents(chunks)

        logger.info("Successfully indexed %d chunk(s) into collection '%s'", len(chunks), self.settings.QDRANT_COLLECTION_NAME)
        return len(chunks)

    def delete_document(self, filename: str) -> bool:
        """
        Deletes all vector points associated with a specific document/source from Qdrant.
        """
        vs = self.vector_store
        if vs is None:
            logger.warning("Vector store unavailable; cannot delete vector points for '%s'", filename)
            return False

        try:
            from qdrant_client.http import models

            filter_selector = models.Filter(
                should=[
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=filename)),
                    models.FieldCondition(key="source", match=models.MatchValue(value=filename)),
                ]
            )

            result = vs.client.delete(
                collection_name=self.settings.QDRANT_COLLECTION_NAME,
                points_selector=filter_selector,
                wait=True,
            )
            logger.info("Deleted Qdrant vector points for source '%s': status=%s", filename, getattr(result, "status", "ok"))
            return True
        except Exception as e:
            logger.error("Failed to delete Qdrant vector points for source '%s': %s", filename, e)
            return False

    def health_check(self) -> Dict[str, Any]:
        """Checks the readiness of the vector store without forcing heavy file locks."""
        has_dir = os.path.exists(self.settings.QDRANT_PATH)
        is_connected = self._vector_store is not None
        return {
            "qdrant_path": self.settings.QDRANT_PATH,
            "collection_name": self.settings.QDRANT_COLLECTION_NAME,
            "storage_path_exists": has_dir,
            "connected": is_connected or has_dir,
        }


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Returns the application RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

"""
Document Ingestion Pipeline for Synk Corp Knowledge Base.
Processes source documents, creates chunked representations, generates vector
embeddings, and indexes them into persistent Qdrant storage.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("synkcorp.ingest")


def load_documents_from_directory(doc_dir: Path) -> List[Document]:
    """Loads all supported text documents from the documents directory."""
    if not doc_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {doc_dir.resolve()}")

    supported_extensions = [".txt", ".md"]
    loaded_docs: List[Document] = []

    for file_path in doc_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            logger.info("Loading document: %s", file_path.name)
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_path.name
                    doc.metadata["file_path"] = str(file_path)
                loaded_docs.extend(docs)
            except Exception as e:
                logger.error("Failed to load %s: %s", file_path.name, e)

    if not loaded_docs:
        raise ValueError(f"No valid documents found in {doc_dir.resolve()}")

    return loaded_docs


def ingest_knowledge_base() -> int:
    """
    Executes the ingestion pipeline:
    1. Loads documents
    2. Chunks text with overlap
    3. Generates embeddings
    4. Indexes vectors into Qdrant
    """
    settings = get_settings()
    docs_path = Path("documents")

    logger.info("Starting document ingestion from '%s'", docs_path.resolve())
    start_time = time.perf_counter()

    # 1. Load documents
    raw_documents = load_documents_from_directory(docs_path)
    logger.info("Loaded %d source document(s)", len(raw_documents))

    # 2. Split into semantic chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)
    logger.info("Generated %d chunk(s) from source documents", len(chunks))

    # 3. Initialize local HuggingFace embeddings
    logger.info("Initializing embedding model '%s'", settings.EMBEDDING_MODEL_NAME)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Store vectors in Qdrant
    logger.info(
        "Indexing vectors into collection '%s' at '%s'",
        settings.QDRANT_COLLECTION_NAME,
        settings.QDRANT_PATH,
    )
    try:
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            path=settings.QDRANT_PATH,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            force_recreate=True,
        )
    except Exception as e:
        logger.error(
            "Failed to write to Qdrant storage. "
            "If the API or another process is running, stop it first to release the lock: %s",
            e,
        )
        raise

    duration = round(time.perf_counter() - start_time, 2)
    logger.info("Successfully indexed %d chunks in %s seconds.", len(chunks), duration)
    return len(chunks)


if __name__ == "__main__":
    try:
        count = ingest_knowledge_base()
        print(f"\n[OK] Ingestion complete. {count} chunks indexed.")
    except Exception as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

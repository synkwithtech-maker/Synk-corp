"""
RAG and knowledge base ingestion routes.
"""

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.rag import (
    RAGUploadResponse,
    UpdateDocumentResponse,
    DeleteDocumentResponse,
    RAGBatchUploadResponse,
    RAGQueryTestRequest,
    RetrievedChunk,
    RAGQueryTestResponse,
    DocumentMetadata,
    DocumentListResponse,
    IngestUrlRequest,
    IngestResponse,
)
from app.services.rag import get_rag_service
from app.services.ingest import load_document_file, load_url

logger = logging.getLogger("synkcorp.api.rag")
router = APIRouter(tags=["RAG Document Processing"])
settings = get_settings()

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".csv", ".json"]


@router.post(
    "/rag/upload",
    summary="Upload New Document for RAG",
    response_model=RAGUploadResponse,
)
async def rag_upload_file(file: UploadFile = File(...)):
    """
    Uploads a new file specifically for RAG purposes:
    1. Saves the file to `documents/`.
    2. Parses text (PDF, DOCX, CSV, JSON, TXT, MD).
    3. Splits text into overlapping semantic chunks.
    4. Generates vector embeddings and indexes them into Qdrant.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    clean_name = Path(filename).name
    ext = Path(clean_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    docs_dir = settings.DOCUMENTS_DIR
    docs_dir.mkdir(exist_ok=True)
    destination = docs_dir / clean_name

    try:
        content = await file.read()
        file_size_kb = round(len(content) / 1024, 2)

        with open(destination, "wb") as f:
            f.write(content)

        logger.info("Saved RAG document '%s' (%s KB)", clean_name, file_size_kb)

        docs = await asyncio.to_thread(load_document_file, destination)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No readable text or records could be extracted from '{clean_name}'.",
            )

        rag_service = get_rag_service()
        chunks_count = await asyncio.to_thread(rag_service.add_documents, docs)
        preview = [doc.page_content[:200] + "..." for doc in docs[:3]]

        return RAGUploadResponse(
            status="success",
            filename=clean_name,
            file_type=ext,
            file_size_kb=file_size_kb,
            chunks_indexed=chunks_count,
            preview_chunks=preview,
            ready_for_chat=True,
            message=f"File '{clean_name}' successfully parsed, vectorized, and indexed into Qdrant for RAG.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process RAG file upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG file ingestion failed: {str(e)}",
        ) from e


@router.put(
    "/rag/documents/{filename}",
    summary="Update / Replace Existing RAG Document",
    response_model=UpdateDocumentResponse,
)
async def update_rag_document(
    filename: str,
    file: UploadFile = File(...),
):
    """
    Updates or replaces an existing corporate document in the RAG knowledge base:
    1. Removes all existing vector chunks associated with this filename from Qdrant.
    2. Overwrites the file in the `documents/` directory with the new version.
    3. Re-parses and re-indexes the new content into Qdrant vectors.
    """
    clean_name = Path(filename).name
    ext = Path(clean_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    docs_dir = settings.DOCUMENTS_DIR
    docs_dir.mkdir(exist_ok=True)
    destination = docs_dir / clean_name

    rag_service = get_rag_service()

    try:
        logger.info("Removing existing vector chunks for document '%s'", clean_name)
        old_vectors_removed = await asyncio.to_thread(rag_service.delete_document, clean_name)

        content = await file.read()
        file_size_kb = round(len(content) / 1024, 2)
        with open(destination, "wb") as f:
            f.write(content)

        logger.info("Overwrote file '%s' on disk (%s KB)", clean_name, file_size_kb)

        docs = await asyncio.to_thread(load_document_file, destination)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded replacement file '{clean_name}' has no readable text content.",
            )

        new_chunks_count = await asyncio.to_thread(rag_service.add_documents, docs)
        preview = [d.page_content[:200] + "..." for d in docs[:3]]

        return UpdateDocumentResponse(
            status="success",
            filename=clean_name,
            file_type=ext,
            file_size_kb=file_size_kb,
            old_vectors_removed=old_vectors_removed,
            new_chunks_indexed=new_chunks_count,
            preview_chunks=preview,
            ready_for_chat=True,
            message=f"Document '{clean_name}' successfully updated. Old vectors removed and {new_chunks_count} new chunks indexed.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update RAG document '%s': %s", clean_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document update failed: {str(e)}",
        ) from e


@router.delete(
    "/rag/documents/{filename}",
    summary="Delete Document from RAG Knowledge Base",
    response_model=DeleteDocumentResponse,
)
async def delete_rag_document(filename: str):
    """
    Deletes a company document from both disk and the Qdrant vector database:
    1. Removes all vector chunks for this source from Qdrant so the AI stops using it.
    2. Deletes the physical file from the `documents/` folder.
    """
    clean_name = Path(filename).name
    docs_dir = settings.DOCUMENTS_DIR
    target_path = docs_dir / clean_name

    rag_service = get_rag_service()
    vectors_deleted = await asyncio.to_thread(rag_service.delete_document, clean_name)

    file_deleted = False
    if target_path.exists() and target_path.is_file():
        try:
            target_path.unlink()
            file_deleted = True
            logger.info("Deleted physical file '%s' from documents/ folder", clean_name)
        except Exception as e:
            logger.warning("Failed to delete physical file '%s': %s", clean_name, e)

    if not vectors_deleted and not file_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{clean_name}' was not found in knowledge base or storage.",
        )

    return DeleteDocumentResponse(
        status="success",
        filename=clean_name,
        file_deleted=file_deleted,
        vectors_deleted=vectors_deleted,
        message=f"Document '{clean_name}' and its vector embeddings were successfully deleted from RAG knowledge base.",
    )


@router.post(
    "/rag/upload-batch",
    summary="Batch Upload Files for RAG",
    response_model=RAGBatchUploadResponse,
)
async def rag_upload_batch(files: List[UploadFile] = File(...)):
    """
    Batch endpoint to upload multiple files at once for RAG processing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    docs_dir = settings.DOCUMENTS_DIR
    docs_dir.mkdir(exist_ok=True)
    rag_service = get_rag_service()

    total_chunks = 0
    results: List[Dict[str, Any]] = []

    for file in files:
        raw_filename = file.filename or "unnamed_file"
        clean_name = Path(raw_filename).name
        ext = Path(clean_name).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": clean_name,
                "status": "skipped",
                "reason": f"Unsupported format '{ext}'",
                "chunks_indexed": 0,
            })
            continue

        destination = docs_dir / clean_name
        try:
            content = await file.read()
            with open(destination, "wb") as f:
                f.write(content)

            docs = await asyncio.to_thread(load_document_file, destination)
            chunks = await asyncio.to_thread(rag_service.add_documents, docs) if docs else 0
            total_chunks += chunks

            results.append({
                "filename": clean_name,
                "status": "success",
                "chunks_indexed": chunks,
                "file_size_kb": round(len(content) / 1024, 2),
            })
        except Exception as e:
            results.append({
                "filename": clean_name,
                "status": "failed",
                "error": str(e),
                "chunks_indexed": 0,
            })

    return RAGBatchUploadResponse(
        status="completed",
        files_processed=len(files),
        total_chunks_indexed=total_chunks,
        results=results,
        message=f"Batch RAG processing completed. {total_chunks} chunks indexed across {len(files)} file(s).",
    )


@router.post(
    "/rag/test-retrieval",
    summary="Test RAG Vector Retrieval (Search Without LLM)",
    response_model=RAGQueryTestResponse,
)
async def rag_test_retrieval(request: RAGQueryTestRequest):
    """
    Tests vector search retrieval against indexed documents without invoking the LLM.
    """
    query = request.query.strip()
    top_k = request.top_k

    rag_service = get_rag_service()
    docs = await asyncio.to_thread(rag_service.retrieve_context, query, top_k)

    chunks = [
        RetrievedChunk(
            source=doc.metadata.get("source", "Unknown"),
            page_content=doc.page_content,
            metadata=doc.metadata,
        )
        for doc in docs
    ]

    return RAGQueryTestResponse(
        query=query,
        total_chunks_found=len(chunks),
        chunks=chunks,
    )


@router.get(
    "/rag/documents",
    summary="List All Indexed RAG Documents",
    response_model=DocumentListResponse,
)
def list_rag_documents():
    """
    Lists all documents currently indexed in the RAG knowledge base directory.
    """
    docs_dir = settings.DOCUMENTS_DIR
    if not docs_dir.exists():
        return DocumentListResponse(total_files=0, documents=[])

    items: List[DocumentMetadata] = []
    for f in docs_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            stats = f.stat()
            modified_iso = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat()
            items.append(
                DocumentMetadata(
                    filename=f.name,
                    file_type=f.suffix.lower(),
                    size_kb=round(stats.st_size / 1024, 2),
                    last_modified=modified_iso,
                )
            )

    return DocumentListResponse(
        total_files=len(items),
        documents=items,
    )


@router.post(
    "/admin/ingest/url",
    summary="Scrape & Ingest Webpage (URL)",
    response_model=IngestResponse,
    tags=["Knowledge Base Ingestion"],
)
async def ingest_webpage(request: IngestUrlRequest):
    """
    Fetches clean text from a live website URL, chunks it, and indexes it into Qdrant.
    """
    url = request.url.strip()
    try:
        logger.info("Scraping webpage for knowledge base: %s", url)
        docs = await asyncio.to_thread(load_url, url)

        rag_service = get_rag_service()
        chunks_count = await asyncio.to_thread(rag_service.add_documents, docs)

        return IngestResponse(
            status="success",
            source=url,
            chunks_indexed=chunks_count,
            message=f"Successfully scraped '{url}' and indexed {chunks_count} chunks into knowledge base.",
        )

    except Exception as e:
        logger.exception("Failed to ingest URL '%s': %s", url, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to crawl or index URL: {str(e)}",
        ) from e

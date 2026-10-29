"""
FastAPI Application for Synk Corp Customer Support.
Features Redis conversation memory, Qdrant-backed RAG, Defense-in-Depth Guardrails,
Real-Time Server-Sent Events (SSE) Token Streaming, Multi-Source Ingestion (PDF, DOCX, CSV, JSON, URL, Text),
Document Lifecycle Management (Upload, Update, Delete, Search), asynchronous execution,
comprehensive health checks, and OpenAPI documentation.
"""

import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import get_settings
from chatbot import ask_question_detailed, stream_chat_events
from memory import get_memory_manager
from rag import get_rag_service
from guardrails import get_guardrails_service
from ingest import load_document_file, load_url, load_documents_from_directory

# ---------------------------------------------------------------------------
# 1. Logging Configuration
# ---------------------------------------------------------------------------

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("synkcorp.api")


# ---------------------------------------------------------------------------
# 2. Lifespan Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan hook: executes startup initializations and
    graceful shutdown resource cleanup.
    """
    logger.info("Starting %s v%s in %s environment", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)

    # Health probe on startup
    memory_manager = get_memory_manager()
    memory_health = memory_manager.health_check()
    logger.info("Memory backend initialized: %s", memory_health)

    rag_service = get_rag_service()
    rag_health = rag_service.health_check()
    logger.info("RAG vector store status: %s", rag_health)

    logger.info(
        "Guardrails active: PromptInjection=%s, MaskPII=%s, StrictGrounding=%s",
        settings.GUARDRAIL_BLOCK_PROMPT_INJECTION,
        settings.GUARDRAIL_MASK_PII,
        settings.GUARDRAIL_STRICT_GROUNDING,
    )

    yield

    logger.info("Shutting down %s. Cleaning up resources...", settings.APP_NAME)


# ---------------------------------------------------------------------------
# 3. Create FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Customer Support AI Engine powered by LangGraph, Qdrant RAG, Redis Memory, AI Safety Guardrails, Real-Time SSE Token Streaming, and Complete RAG File Lifecycle Management (Upload, Update, Delete, Search).",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for external frontends and web widgets
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 4. Request & Response Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's question or message",
        examples=["What services does Synk Corp provide?"],
    )
    session_id: str = Field(
        default="default-session",
        min_length=1,
        max_length=128,
        description="Unique session identifier for Redis conversation tracking",
        examples=["customer-session-123"],
    )


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list, description="Documents referenced for the answer")
    guardrail_passed: bool = Field(default=True, description="Whether query passed safety guardrails")
    guardrail_category: str = Field(default="clean", description="Safety category: clean, pii_masked, prompt_injection, system_leak_prevented")
    latency_ms: float = Field(..., description="Request processing time in milliseconds")


class DialogueMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    history: List[DialogueMessage]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    app_version: str
    environment: str
    model: str
    memory: Dict[str, Any]
    rag: Dict[str, Any]
    guardrails: Dict[str, Any]
    features: List[str]


class IngestUrlRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=8,
        description="Target webpage URL to crawl and index",
        examples=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    )


class IngestResponse(BaseModel):
    status: str
    source: str
    chunks_indexed: int
    message: str


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    size_kb: float
    last_modified: str


class DocumentListResponse(BaseModel):
    total_files: int
    documents: List[DocumentMetadata]


# RAG File Management Schemas
class RAGUploadResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    file_size_kb: float
    chunks_indexed: int
    preview_chunks: List[str]
    ready_for_chat: bool
    message: str


class UpdateDocumentResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    file_size_kb: float
    old_vectors_removed: bool
    new_chunks_indexed: int
    preview_chunks: List[str]
    ready_for_chat: bool
    message: str


class DeleteDocumentResponse(BaseModel):
    status: str
    filename: str
    file_deleted: bool
    vectors_deleted: bool
    message: str


class RAGBatchUploadResponse(BaseModel):
    status: str
    files_processed: int
    total_chunks_indexed: int
    results: List[Dict[str, Any]]
    message: str


class RAGQueryTestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Test search query against knowledge base", examples=["refund policy"])
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")


class RetrievedChunk(BaseModel):
    source: str
    page_content: str
    metadata: Dict[str, Any]


class RAGQueryTestResponse(BaseModel):
    query: str
    total_chunks_found: int
    chunks: List[RetrievedChunk]


# ---------------------------------------------------------------------------
# 5. Core API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Root Status",
    response_model=Dict[str, Any],
    tags=["General"],
)
def home():
    """Returns application name and operational status with direct links."""
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.APP_VERSION,
        "swagger_docs": "/docs",
        "redoc_docs": "/redoc",
        "openapi_json": "/openapi.json",
        "health": "/health",
        "streaming_endpoint": "/chat/stream",
        "rag_upload_endpoint": "/rag/upload",
    }


@app.get(
    "/health",
    summary="Health & Readiness Probe",
    response_model=HealthResponse,
    tags=["Monitoring"],
)
def health():
    """Checks the status of the API, Redis connection, Qdrant store, and Guardrails."""
    memory_manager = get_memory_manager()
    rag_service = get_rag_service()

    mem_health = memory_manager.health_check()
    rag_health = rag_service.health_check()

    overall_status = "healthy"
    if not rag_health.get("connected", False) and not rag_health.get("storage_path_exists", False):
        overall_status = "degraded"

    guardrails_info = {
        "enabled": settings.GUARDRAILS_ENABLED,
        "block_prompt_injection": settings.GUARDRAIL_BLOCK_PROMPT_INJECTION,
        "mask_pii": settings.GUARDRAIL_MASK_PII,
        "strict_grounding": settings.GUARDRAIL_STRICT_GROUNDING,
    }

    return HealthResponse(
        status=overall_status,
        app_version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        model=settings.OPENROUTER_MODEL,
        memory=mem_health,
        rag=rag_health,
        guardrails=guardrails_info,
        features=[
            "rag_search",
            "redis_conversation_memory",
            "defense_in_depth_guardrails",
            "realtime_sse_streaming",
            "multi_source_ingestion_pdf_docx_csv_json_url",
            "dedicated_rag_upload_api",
            "rag_document_lifecycle_update_delete",
        ],
    )


# Browser helper when someone navigates directly to /chat in address bar
@app.get("/chat", include_in_schema=False)
@app.get("/chat/", include_in_schema=False)
def chat_get_info():
    """Explains how to use the POST /chat endpoint when visited via browser GET."""
    return {
        "status": "info",
        "message": "The /chat endpoint expects a POST request with a JSON body.",
        "example_curl": (
            'curl -X POST "http://127.0.0.1:8000/chat" '
            '-H "Content-Type: application/json" '
            '-d \'{"session_id": "session-1", "question": "What services does Synk Corp provide?"}\''
        ),
        "streaming_alternative": "For real-time streaming, visit GET /chat/stream?question=YourQuestion or POST /chat/stream",
        "docs_url": "/docs#/Chatbot/chat_chat_post",
    }


@app.post(
    "/chat",
    summary="Submit Support Question (Synchronous)",
    response_model=ChatResponse,
    tags=["Chatbot"],
)
@app.post(
    "/chat/",
    include_in_schema=False,
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    """
    Submits a user question for a conversation session.
    Applies input safety guardrails (injection prevention, PII masking),
    retrieves previous dialogue history from Redis, fetches relevant documents
    from Qdrant, verifies output grounding, and returns the response.
    """
    question = request.question.strip()
    session_id = request.session_id.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty or whitespace.",
        )

    start_time = time.perf_counter()

    try:
        logger.info("Processing chat request for session_id='%s'", session_id)

        # Run CPU/network bound LLM and guarded RAG graph in an async worker thread
        result = await asyncio.to_thread(
            ask_question_detailed,
            question=question,
            session_id=session_id,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Completed chat request for session_id='%s' in %s ms (guardrail=%s)",
            session_id,
            latency_ms,
            result.get("guardrail_category", "clean"),
        )

        return ChatResponse(
            session_id=session_id,
            question=result.get("question", question),
            answer=result["answer"],
            sources=result.get("sources", []),
            guardrail_passed=result.get("guardrail_passed", True),
            guardrail_category=result.get("guardrail_category", "clean"),
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.exception("Chat processing failed for session_id='%s'", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the support answer.",
        ) from e


# ---------------------------------------------------------------------------
# 6. Real-Time Token Streaming Endpoints (Server-Sent Events / SSE)
# ---------------------------------------------------------------------------

@app.post(
    "/chat/stream",
    summary="Real-Time Token Streaming (POST SSE)",
    tags=["Streaming"],
    response_description="Server-Sent Events stream of live tokens, citations, and status events.",
)
async def chat_stream_post(request: ChatRequest):
    """
    Streams the assistant's answer chunk-by-chunk in real time using Server-Sent Events (SSE).
    """
    question = request.question.strip()
    session_id = request.session_id.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    logger.info("Initiating POST SSE token stream for session_id='%s'", session_id)

    return StreamingResponse(
        stream_chat_events(question=question, session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/chat/stream",
    summary="Real-Time Token Streaming (GET SSE / Browser EventSource)",
    tags=["Streaming"],
    response_description="Server-Sent Events stream accessible via browser URL or native EventSource.",
)
async def chat_stream_get(
    question: str = Query(..., min_length=1, max_length=2000, description="The customer's question"),
    session_id: str = Query(default="default-session", min_length=1, max_length=128, description="Unique session ID"),
):
    """
    GET version of the SSE streaming endpoint.
    Allows testing directly from your browser address bar or connecting via JavaScript's `new EventSource('/chat/stream?question=...')`.
    """
    clean_q = question.strip()
    clean_sid = session_id.strip()

    logger.info("Initiating GET SSE token stream for session_id='%s'", clean_sid)

    return StreamingResponse(
        stream_chat_events(question=clean_q, session_id=clean_sid),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 7. Complete RAG Document Lifecycle APIs (Upload, Update, Delete, Search)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".csv", ".json"]


@app.post(
    "/rag/upload",
    summary="Upload New Document for RAG",
    response_model=RAGUploadResponse,
    tags=["RAG Document Processing"],
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

    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    destination = docs_dir / clean_name

    try:
        content = await file.read()
        file_size_kb = round(len(content) / 1024, 2)

        with open(destination, "wb") as f:
            f.write(content)

        logger.info("Saved RAG document '%s' (%s KB)", clean_name, file_size_kb)

        # Extract text into Document objects
        docs = await asyncio.to_thread(load_document_file, destination)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No readable text or records could be extracted from '{clean_name}'.",
            )

        # Index into Qdrant
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


@app.put(
    "/rag/documents/{filename}",
    summary="Update / Replace Existing RAG Document",
    response_model=UpdateDocumentResponse,
    tags=["RAG Document Processing"],
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

    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    destination = docs_dir / clean_name

    rag_service = get_rag_service()

    try:
        # Step 1: Remove old vectors for this filename from Qdrant
        logger.info("Removing existing vector chunks for document '%s'", clean_name)
        old_vectors_removed = await asyncio.to_thread(rag_service.delete_document, clean_name)

        # Step 2: Overwrite file on disk
        content = await file.read()
        file_size_kb = round(len(content) / 1024, 2)
        with open(destination, "wb") as f:
            f.write(content)

        logger.info("Overwrote file '%s' on disk (%s KB)", clean_name, file_size_kb)

        # Step 3: Parse and index new content
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


@app.delete(
    "/rag/documents/{filename}",
    summary="Delete Document from RAG Knowledge Base",
    response_model=DeleteDocumentResponse,
    tags=["RAG Document Processing"],
)
async def delete_rag_document(filename: str):
    """
    Deletes a company document from both disk and the Qdrant vector database:
    1. Removes all vector chunks for this source from Qdrant so the AI stops using it.
    2. Deletes the physical file from the `documents/` folder.
    """
    clean_name = Path(filename).name
    docs_dir = Path("documents")
    target_path = docs_dir / clean_name

    rag_service = get_rag_service()

    # Step 1: Delete vectors from Qdrant
    vectors_deleted = await asyncio.to_thread(rag_service.delete_document, clean_name)

    # Step 2: Delete file from disk
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


@app.post(
    "/rag/upload-batch",
    summary="Batch Upload Files for RAG",
    response_model=RAGBatchUploadResponse,
    tags=["RAG Document Processing"],
)
async def rag_upload_batch(files: List[UploadFile] = File(...)):
    """
    Batch endpoint to upload multiple files at once for RAG processing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    docs_dir = Path("documents")
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


@app.post(
    "/rag/test-retrieval",
    summary="Test RAG Vector Retrieval (Search Without LLM)",
    response_model=RAGQueryTestResponse,
    tags=["RAG Document Processing"],
)
async def rag_test_retrieval(request: RAGQueryTestRequest):
    """
    Tests vector search retrieval against indexed documents without invoking the LLM.
    Use this endpoint to verify that uploaded files can be accurately retrieved.
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


@app.get(
    "/rag/documents",
    summary="List All Indexed RAG Documents",
    response_model=DocumentListResponse,
    tags=["RAG Document Processing"],
)
def list_rag_documents():
    """
    Lists all documents currently indexed in the RAG knowledge base directory.
    """
    docs_dir = Path("documents")
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


# ---------------------------------------------------------------------------
# 8. Web Ingestion & Session History Endpoints
# ---------------------------------------------------------------------------

@app.post(
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


@app.get("/chat/history", include_in_schema=False)
@app.get("/chat/history/", include_in_schema=False)
def chat_history_info():
    """Helper when /chat/history is visited without specifying a session_id."""
    return {
        "status": "info",
        "message": "Please include a session_id in the URL: /chat/{session_id}/history",
        "example": "/chat/session-1/history",
    }


@app.get(
    "/chat/{session_id}/history",
    summary="Retrieve Session History",
    response_model=SessionHistoryResponse,
    tags=["Chatbot"],
)
@app.get(
    "/chat/{session_id}/history/",
    include_in_schema=False,
    response_model=SessionHistoryResponse,
)
def get_session_history(session_id: str):
    """
    Fetches the persisted conversation history for a given session from Redis.
    """
    cleaned_id = session_id.strip()
    if not cleaned_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID must not be empty.",
        )

    memory_manager = get_memory_manager()
    raw_history = memory_manager.get_messages(cleaned_id)

    history = [
        DialogueMessage(
            role=msg.get("role", "unknown"),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp"),
            metadata=msg.get("metadata"),
        )
        for msg in raw_history
    ]

    return SessionHistoryResponse(
        session_id=cleaned_id,
        message_count=len(history),
        history=history,
    )


@app.delete(
    "/chat/{session_id}",
    summary="Clear Session Conversation",
    response_model=ClearSessionResponse,
    tags=["Chatbot"],
)
@app.delete(
    "/chat/{session_id}/",
    include_in_schema=False,
    response_model=ClearSessionResponse,
)
def clear_session(session_id: str):
    """
    Deletes the conversation history for a session from Redis.
    """
    cleaned_id = session_id.strip()
    if not cleaned_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID must not be empty.",
        )

    memory_manager = get_memory_manager()
    cleared = memory_manager.clear(cleaned_id)

    return ClearSessionResponse(
        session_id=cleaned_id,
        cleared=cleared,
        message=f"Session history for '{cleaned_id}' cleared successfully."
        if cleared
        else f"No active conversation found for session '{cleaned_id}'.",
    )

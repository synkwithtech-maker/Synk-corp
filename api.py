"""
FastAPI Application for Synk Corp Customer Support.
Features Redis conversation memory, Qdrant-backed RAG, Defense-in-Depth Guardrails,
asynchronous thread execution, comprehensive health checks, and OpenAPI documentation.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings
from chatbot import ask_question_detailed
from memory import get_memory_manager
from rag import get_rag_service
from guardrails import get_guardrails_service

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
    description="Enterprise-grade Customer Support Chatbot powered by LangGraph, Qdrant RAG, Redis Conversation Memory, and AI Safety Guardrails.",
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
    guardrail_category: str = Field(default="clean", description="Safety category: clean, pii_masked, prompt_injection, system_prompt_leak_prevented")
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


# ---------------------------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Root Status",
    response_model=Dict[str, str],
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
        "docs_url": "/docs#/Chatbot/chat_chat_post",
    }


@app.post(
    "/chat",
    summary="Submit Support Question",
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

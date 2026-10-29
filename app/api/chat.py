"""
Chat and streaming API routes.
"""

import asyncio
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.core.security import verify_chat_key, chat_rate_limiter, validate_session_id

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    DialogueMessage,
    SessionHistoryResponse,
    ClearSessionResponse,
)
from app.services.chatbot import ask_question_detailed, stream_chat_events
from app.services.memory import get_memory_manager

logger = logging.getLogger("synkcorp.api.chat")
router = APIRouter(tags=["Chatbot"])


@router.post(
    "/chat",
    summary="Submit Support Question (Synchronous)",
    response_model=ChatResponse,
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
@router.post(
    "/chat/",
    include_in_schema=False,
    response_model=ChatResponse,
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
async def chat(request: ChatRequest):
    """
    Submits a user question for a conversation session.
    Applies input safety guardrails, retrieves history from Redis,
    fetches documents from Qdrant, verifies output grounding, and returns response.
    """
    question = request.question.strip()
    session_id = validate_session_id(request.session_id)

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    start_time = time.perf_counter()

    try:
        logger.info("Processing chat request for session_id='%s'", session_id)
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


@router.post(
    "/chat/stream",
    summary="Real-Time Token Streaming (POST SSE)",
    tags=["Streaming"],
    response_description="Server-Sent Events stream of live tokens, citations, and status events.",
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
async def chat_stream_post(request: ChatRequest):
    """
    Streams the assistant's answer chunk-by-chunk in real time using Server-Sent Events (SSE).
    """
    question = request.question.strip()
    session_id = validate_session_id(request.session_id)

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


@router.get(
    "/chat/stream",
    summary="Real-Time Token Streaming (GET SSE / Browser EventSource)",
    tags=["Streaming"],
    response_description="Server-Sent Events stream accessible via browser URL or native EventSource.",
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
async def chat_stream_get(
    question: str = Query(..., min_length=1, max_length=2000, description="The customer's question"),
    session_id: str = Query(default="default-session", min_length=1, max_length=64, description="Unique session ID"),
):
    """
    GET version of the SSE streaming endpoint.
    Allows testing directly from your browser address bar or connecting via JavaScript's `new EventSource('/chat/stream?question=...')`.
    """
    clean_q = question.strip()
    clean_sid = validate_session_id(session_id)

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


@router.get("/chat", include_in_schema=False)
@router.get("/chat/", include_in_schema=False)
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


@router.get("/chat/history", include_in_schema=False)
@router.get("/chat/history/", include_in_schema=False)
def chat_history_info():
    """Helper when /chat/history is visited without specifying a session_id."""
    return {
        "status": "info",
        "message": "Please include a session_id in the URL: /chat/{session_id}/history",
        "example": "/chat/session-1/history",
    }


@router.get(
    "/chat/{session_id}/history",
    summary="Retrieve Session History",
    response_model=SessionHistoryResponse,
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
@router.get(
    "/chat/{session_id}/history/",
    include_in_schema=False,
    response_model=SessionHistoryResponse,
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
def get_session_history(session_id: str):
    """
    Fetches the persisted conversation history for a given session from Redis.
    """
    cleaned_id = validate_session_id(session_id)

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


@router.delete(
    "/chat/{session_id}",
    summary="Clear Session Conversation",
    response_model=ClearSessionResponse,
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
@router.delete(
    "/chat/{session_id}/",
    include_in_schema=False,
    response_model=ClearSessionResponse,
    dependencies=[Depends(verify_chat_key), Depends(chat_rate_limiter)],
)
def clear_session(session_id: str):
    """
    Deletes the conversation history for a session from Redis.
    """
    cleaned_id = validate_session_id(session_id)

    memory_manager = get_memory_manager()
    cleared = memory_manager.clear(cleaned_id)

    return ClearSessionResponse(
        session_id=cleaned_id,
        cleared=cleared,
        message=f"Session history for '{cleaned_id}' cleared successfully."
        if cleared
        else f"No active conversation found for session '{cleaned_id}'.",
    )

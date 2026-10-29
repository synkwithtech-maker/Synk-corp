"""
Health check and root API routes.
"""

from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.memory import get_memory_manager
from app.services.rag import get_rag_service

router = APIRouter(tags=["Monitoring"])
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    app_version: str
    environment: str
    model: str
    memory: Dict[str, Any]
    rag: Dict[str, Any]
    guardrails: Dict[str, Any]
    features: list[str]


@router.get(
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


@router.get(
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

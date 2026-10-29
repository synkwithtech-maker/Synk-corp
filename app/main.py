"""
Main FastAPI Application Entrypoint for Synk Corp Support Bot.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.security import SecurityHeadersMiddleware
from app.services.memory import get_memory_manager
from app.services.rag import get_rag_service
from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.rag import router as rag_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("synkcorp.app")


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


def create_app() -> FastAPI:
    """Application factory for Synk Corp Support API."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Enterprise Customer Support AI Engine powered by LangGraph, Qdrant RAG, Redis Memory, AI Safety Guardrails, Real-Time SSE Token Streaming, and Complete RAG File Lifecycle Management.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. Apply OWASP Security Headers (Nosniff, Deny iframe clickjacking, XSS, Referrer)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Hardened CORS configuration
    cors_origins = settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["*"]
    cors_allow_creds = True if "*" not in cors_origins else False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include modular route controllers directly
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(rag_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

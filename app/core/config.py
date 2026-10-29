"""
Configuration management for Synk Corp Support Bot.
Uses Pydantic Settings for strict typing, default values, and .env loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    APP_NAME: str = Field(default="Synk Corp Support API", description="Application name")
    APP_VERSION: str = Field(default="2.2.0", description="Application version")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # OpenRouter / LLM settings
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, description="OpenRouter API key")
    OPENROUTER_MODEL: str = Field(
        default="qwen/qwen3.8-27b",
        description="LLM model identifier on OpenRouter",
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base endpoint",
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="Direct OpenAI API key (optional fallback)")
    TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0, description="Model generation temperature")
    MAX_TOKENS: int = Field(default=1000, ge=50, le=8000, description="Max generation tokens")

    # Embedding & Vector Database settings
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for embeddings",
    )
    QDRANT_PATH: str = Field(default=str(ROOT_DIR / "qdrant_data"), description="Local path for Qdrant storage")
    QDRANT_COLLECTION_NAME: str = Field(default="synk_company_docs", description="Qdrant collection name")
    TOP_K_DOCUMENTS: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve")

    # Storage paths
    DOCUMENTS_DIR: Path = Field(default=ROOT_DIR / "documents", description="Folder storing knowledge documents")

    # Redis conversation memory settings
    REDIS_ENABLED: bool = Field(default=True, description="Enable Redis memory backend")
    REDIS_URL: Optional[str] = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (takes precedence over host/port if set)",
    )
    REDIS_HOST: str = Field(default="localhost", description="Redis server host")
    REDIS_PORT: int = Field(default=6379, description="Redis server port")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password if protected")
    REDIS_DB: int = Field(default=0, ge=0, description="Redis database index")
    REDIS_KEY_PREFIX: str = Field(default="synk:chat:session", description="Key namespace prefix in Redis")
    REDIS_TTL_SECONDS: int = Field(
        default=86400,  # 24 hours
        ge=60,
        description="Conversation session time-to-live in seconds",
    )
    MAX_HISTORY_TURNS: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Maximum recent dialogue turns to pass to prompt context",
    )

    # Security & Guardrails settings
    GUARDRAILS_ENABLED: bool = Field(default=True, description="Master switch for safety guardrails")
    GUARDRAIL_BLOCK_PROMPT_INJECTION: bool = Field(
        default=True,
        description="Detect and block prompt injections, jailbreaks, and system overrides",
    )
    GUARDRAIL_MASK_PII: bool = Field(
        default=True,
        description="Mask sensitive PII (credit cards, SSNs, API tokens) before storage",
    )
    GUARDRAIL_STRICT_GROUNDING: bool = Field(
        default=True,
        description="Verify outputs adhere strictly to company document context",
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of application settings."""
    return Settings()

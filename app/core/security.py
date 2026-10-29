"""
Security, Authentication, Rate Limiting, and Anti-Abuse Protections for Synk Corp API.
"""

from collections import defaultdict, deque
import ipaddress
import logging
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from typing import Optional
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger("synkcorp.security")
settings = get_settings()

# Security schemes for OpenAPI / Swagger UI Authorize dialog
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, description="Admin or Client API Key")
bearer_token_scheme = HTTPBearer(auto_error=False, description="Bearer token authorization")

# Windows reserved device names to block
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


# =====================================================================
# 1. API Key Authentication & Verification
# =====================================================================

def extract_api_key(
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_token_scheme),
) -> Optional[str]:
    """Extracts API key from either X-API-Key header or Authorization: Bearer token."""
    if header_key and header_key.strip():
        return header_key.strip()
    if bearer_creds and bearer_creds.credentials and bearer_creds.credentials.strip():
        return bearer_creds.credentials.strip()
    return None


def verify_admin_key(api_key: Optional[str] = Depends(extract_api_key)) -> bool:
    """
    Guards administrative and destructive RAG lifecycle endpoints.
    Requires a valid ADMIN_API_KEY.
    """
    if not api_key:
        logger.warning("Unauthorized admin access attempt: No API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide 'X-API-Key' or 'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    is_valid = secrets.compare_digest(api_key, settings.ADMIN_API_KEY)
    if not is_valid:
        logger.warning("Forbidden admin access attempt with invalid key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Admin API Key. Administrative action forbidden.",
        )

    return True


def verify_chat_key(api_key: Optional[str] = Depends(extract_api_key)) -> bool:
    """
    Protects user chat and streaming endpoints when REQUIRE_API_KEY_FOR_CHAT is enabled.
    Can be disabled for open public website widgets via config.
    """
    if not settings.REQUIRE_API_KEY_FOR_CHAT:
        return True

    if not api_key:
        logger.warning("Unauthorized chat access attempt: No API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chat authentication required. Provide 'X-API-Key' or 'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    valid_keys = [settings.ADMIN_API_KEY]
    if settings.CHAT_API_KEY:
        valid_keys.append(settings.CHAT_API_KEY)

    is_valid = any(secrets.compare_digest(api_key, expected) for expected in valid_keys)
    if not is_valid:
        logger.warning("Forbidden chat access attempt with invalid key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Client API Key for chat access.",
        )

    return True


# =====================================================================
# 2. Rate Limiting (Redis-backed with In-Memory Sliding Window Fallback)
# =====================================================================

class RateLimiter:
    """
    Enforces per-client IP sliding window rate limiting.
    Protects against DDoS attacks and LLM API wallet exhaustion.
    """

    def __init__(self, requests_per_minute: int, namespace: str = "general"):
        self.limit = requests_per_minute
        self.namespace = namespace
        self._memory_lock = threading.Lock()
        self._memory_buckets: dict[str, deque] = defaultdict(deque)

    def _get_client_ip(self, request: Request) -> str:
        """Extracts client IP, respecting forward proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def __call__(self, request: Request) -> None:
        client_ip = self._get_client_ip(request)
        now = time.time()
        window_seconds = 60

        # 1. Attempt Redis sliding window if available
        from app.services.memory import get_memory_manager
        memory_manager = get_memory_manager()

        if getattr(memory_manager, "is_redis", False) and hasattr(memory_manager, "_redis"):
            try:
                redis_client = memory_manager._redis
                key = f"synk:ratelimit:{self.namespace}:{client_ip}"
                current_count = redis_client.incr(key)
                if current_count == 1:
                    redis_client.expire(key, window_seconds)

                if current_count > self.limit:
                    ttl = redis_client.ttl(key)
                    retry_after = max(1, ttl if ttl > 0 else window_seconds)
                    logger.warning("Rate limit exceeded for IP %s on '%s' (Redis)", client_ip, self.namespace)
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Maximum {self.limit} requests per minute for this endpoint.",
                        headers={"Retry-After": str(retry_after)},
                    )
                return
            except HTTPException:
                raise
            except Exception as e:
                logger.debug("Redis rate limiter failed, falling back to memory: %s", e)

        # 2. In-memory sliding window fallback
        with self._memory_lock:
            bucket = self._memory_buckets[client_ip]
            cutoff = now - window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])))
                logger.warning("Rate limit exceeded for IP %s on '%s' (Memory)", client_ip, self.namespace)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {self.limit} requests per minute for this endpoint.",
                    headers={"Retry-After": str(retry_after)},
                )

            bucket.append(now)


# Standard rate limiters
chat_rate_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_CHAT_PER_MINUTE, namespace="chat")
admin_rate_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_ADMIN_PER_MINUTE, namespace="admin")


# =====================================================================
# 3. Server-Side Request Forgery (SSRF) Protection
# =====================================================================

def validate_safe_url(url: str) -> str:
    """
    Validates that a URL does not point to internal resources, loopback,
    private LANs, or cloud metadata services (e.g. 169.254.169.254).
    """
    clean_url = url.strip()
    try:
        parsed = urlparse(clean_url)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed URL: {str(e)}",
        )

    # Scheme restriction
    if parsed.scheme.lower() not in ["http", "https"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid URL protocol '{parsed.scheme}'. Only 'http' and 'https' are allowed.",
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must specify a valid target hostname.",
        )

    # Disallow localhost keywords directly
    if hostname.lower() in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to localhost or loopback destinations is strictly forbidden.",
        )

    # DNS resolution & IP check
    try:
        resolved_ips = socket.getaddrinfo(hostname, parsed.port or 80)
        for entry in resolved_ips:
            ip_str = entry[4][0]
            ip_obj = ipaddress.ip_address(ip_str)

            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_reserved
                or ip_obj.is_multicast
            ):
                logger.warning("SSRF block: Hostname '%s' resolved to forbidden IP '%s'", hostname, ip_str)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden target: Destination '{hostname}' resolves to an internal or private network address.",
                )
    except HTTPException:
        raise
    except socket.gaierror:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not resolve host '{hostname}'. Ensure domain name is valid.",
        )
    except Exception as e:
        logger.error("Error during SSRF validation for '%s': %s", url, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL address validation failed.",
        )

    return clean_url


# =====================================================================
# 4. Filename Sanitization & Path Traversal Prevention
# =====================================================================

def sanitize_and_validate_filename(raw_filename: str, allowed_extensions: list[str]) -> str:
    """
    Sanitizes filenames to prevent path traversal, null byte injections,
    and OS-specific device collisions.
    """
    if not raw_filename or not raw_filename.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename cannot be empty.")

    # Strip path separators, null bytes, and traversal elements
    basename = Path(raw_filename).name.strip().replace("\x00", "")

    # Restrict to safe characters: alphanumeric, dashes, underscores, dots
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", basename)

    # Disallow leading dots (hidden files) or starting with hyphens
    clean_name = clean_name.lstrip(".-")
    if not clean_name:
        clean_name = "uploaded_document"

    # Enforce reasonable length
    if len(clean_name) > 128:
        base, ext = Path(clean_name).stem[:100], Path(clean_name).suffix
        clean_name = f"{base}{ext}"

    # Check extension
    ext = Path(clean_name).suffix.lower()
    if ext not in [e.lower() for e in allowed_extensions]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension '{ext}' is forbidden. Allowed: {', '.join(allowed_extensions)}",
        )

    # Block Windows reserved device names
    stem = Path(clean_name).stem.upper()
    if stem in WINDOWS_RESERVED_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filename stem '{stem}' is a reserved system identifier.",
        )

    return clean_name


def validate_file_size(content: bytes, max_mb: int = settings.MAX_UPLOAD_SIZE_MB) -> None:
    """Rejects payload if file size exceeds allowed threshold."""
    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        size_mb = round(len(content) / (1024 * 1024), 2)
        logger.warning("Upload rejected: File size %s MB exceeds limit %s MB", size_mb, max_mb)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb} MB) exceeds maximum allowed upload limit of {max_mb} MB.",
        )


# =====================================================================
# 5. Session ID Validation
# =====================================================================

def validate_session_id(session_id: str) -> str:
    """Validates session_id against injection attacks and malformed formats."""
    clean_id = session_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", clean_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id. Must be 1-64 characters alphanumeric, underscores, or hyphens.",
        )
    return clean_id


# =====================================================================
# 6. HTTP Security Headers Middleware (OWASP Defense)
# =====================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Applies recommended OWASP HTTP response security headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Content sniffing defense
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Clickjacking defense
        response.headers["X-Frame-Options"] = "DENY"

        # Legacy XSS filter defense
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer privacy defense
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Server identification reduction
        if "server" in response.headers:
            del response.headers["server"]

        return response

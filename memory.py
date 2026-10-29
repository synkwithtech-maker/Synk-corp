"""
Production-grade Conversation Memory Module.
Supports Redis storage with TTL and sliding windows, with a resilient
in-memory fallback when Redis is unreachable.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import redis
from config import Settings, get_settings

logger = logging.getLogger(__name__)


class BaseMemoryManager:
    """Base interface for conversation memory managers."""

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def clear(self, session_id: str) -> bool:
        raise NotImplementedError

    def health_check(self) -> Dict[str, Any]:
        raise NotImplementedError


class RedisMemoryManager(BaseMemoryManager):
    """
    Redis-backed conversation memory manager.
    Stores messages in a Redis List under a namespaced key, with sliding TTL.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.prefix = settings.REDIS_KEY_PREFIX
        self.ttl = settings.REDIS_TTL_SECONDS
        self._client: Optional[redis.Redis] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initializes Redis client with connection pooling and timeouts."""
        try:
            if self.settings.REDIS_URL:
                self._client = redis.Redis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                )
            else:
                self._client = redis.Redis(
                    host=self.settings.REDIS_HOST,
                    port=self.settings.REDIS_PORT,
                    password=self.settings.REDIS_PASSWORD,
                    db=self.settings.REDIS_DB,
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                )
            # Test connection
            self._client.ping()
            logger.info("Connected to Redis conversation store at %s", self.settings.REDIS_URL or self.settings.REDIS_HOST)
        except Exception as e:
            logger.warning("Failed to connect to Redis: %s. Operations will fall back gracefully.", e)
            self._client = None

    def _get_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:messages"

    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves messages for a session.
        If limit is provided, returns the most recent 'limit' messages.
        """
        if not self.is_connected():
            return []

        key = self._get_key(session_id)
        try:
            # Redis list stores messages chronologically (RPUSH)
            raw_items = self._client.lrange(key, 0, -1)
            messages = [json.loads(item) for item in raw_items]

            if limit and limit > 0:
                return messages[-limit:]
            return messages
        except Exception as e:
            logger.error("Error reading conversation from Redis for session '%s': %s", session_id, e)
            return []

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Appends user and assistant messages atomically and refreshes TTL.
        """
        if not self.is_connected():
            logger.warning("Redis client not connected; skipping persistence for session '%s'", session_id)
            return

        key = self._get_key(session_id)
        timestamp = datetime.now(timezone.utc).isoformat()

        user_entry = json.dumps({
            "role": "user",
            "content": user_message,
            "timestamp": timestamp,
        })

        assistant_entry = json.dumps({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": timestamp,
            "metadata": metadata or {},
        })

        try:
            pipeline = self._client.pipeline()
            pipeline.rpush(key, user_entry, assistant_entry)
            # Keep max 50 turns (100 messages) per session to prevent unbounded memory growth
            pipeline.ltrim(key, -100, -1)
            pipeline.expire(key, self.ttl)
            pipeline.execute()
            logger.debug("Appended dialogue turn to Redis for session: %s", session_id)
        except Exception as e:
            logger.error("Failed to append dialogue turn to Redis for session '%s': %s", session_id, e)

    def clear(self, session_id: str) -> bool:
        """Deletes all messages for the specified session."""
        if not self.is_connected():
            return False
        key = self._get_key(session_id)
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            logger.error("Failed to clear session '%s' in Redis: %s", session_id, e)
            return False

    def health_check(self) -> Dict[str, Any]:
        """Performs a Redis health probe measuring latency."""
        start = time.perf_counter()
        try:
            if self._client and self._client.ping():
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return {
                    "backend": "redis",
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "connected": True,
                }
        except Exception as e:
            return {
                "backend": "redis",
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }
        return {
            "backend": "redis",
            "status": "disconnected",
            "connected": False,
        }


class InMemoryMemoryManager(BaseMemoryManager):
    """
    In-memory fallback conversation memory store.
    Used when Redis is not configured or temporarily unreachable.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        history = self._store.get(session_id, [])
        if limit and limit > 0:
            return history[-limit:]
        return list(history)

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if session_id not in self._store:
            self._store[session_id] = []

        timestamp = datetime.now(timezone.utc).isoformat()
        self._store[session_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": timestamp,
        })
        self._store[session_id].append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": timestamp,
            "metadata": metadata or {},
        })
        # Keep recent 100 messages
        self._store[session_id] = self._store[session_id][-100:]

    def clear(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def health_check(self) -> Dict[str, Any]:
        return {
            "backend": "in-memory-fallback",
            "status": "healthy",
            "sessions_count": len(self._store),
            "connected": True,
        }


class ResilientMemoryService(BaseMemoryManager):
    """
    Composite memory manager that delegates to Redis when available,
    and seamlessly falls back to InMemory storage if Redis is down.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.redis_manager = RedisMemoryManager(self.settings) if self.settings.REDIS_ENABLED else None
        self.fallback_manager = InMemoryMemoryManager(self.settings)

    @property
    def active_backend(self) -> BaseMemoryManager:
        if self.redis_manager and self.redis_manager.is_connected():
            return self.redis_manager
        return self.fallback_manager

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.active_backend.get_messages(session_id, limit=limit)

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.active_backend.append_turn(session_id, user_message, assistant_message, metadata)

    def clear(self, session_id: str) -> bool:
        # Clear in both to maintain consistency
        redis_cleared = self.redis_manager.clear(session_id) if self.redis_manager else False
        fallback_cleared = self.fallback_manager.clear(session_id)
        return redis_cleared or fallback_cleared

    def health_check(self) -> Dict[str, Any]:
        backend_info = self.active_backend.health_check()
        backend_info["redis_enabled"] = self.settings.REDIS_ENABLED
        if self.redis_manager:
            backend_info["redis_ping"] = self.redis_manager.health_check()
        return backend_info


# Global singleton instance
_memory_service: Optional[ResilientMemoryService] = None


def get_memory_manager() -> ResilientMemoryService:
    """Returns the application memory service singleton."""
    global _memory_service
    if _memory_service is None:
        _memory_service = ResilientMemoryService()
    return _memory_service

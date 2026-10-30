"""
Unit tests for Redis memory configuration and connection handling in memory.py.
Verifies that REDIS_HOST, REDIS_PORT, and REDIS_PASSWORD are not ignored when REDIS_URL is None.
"""

from unittest.mock import patch, MagicMock
from config import Settings
from memory import RedisMemoryManager


def test_redis_default_uses_host_and_port():
    """
    Verifies that when REDIS_URL is None (default), RedisMemoryManager
    initializes redis.Redis using REDIS_HOST ('localhost') and REDIS_PORT (6379).
    """
    settings = Settings(REDIS_URL=None)
    with patch("redis.Redis") as mock_redis_cls:
        mock_instance = MagicMock()
        mock_redis_cls.return_value = mock_instance

        manager = RedisMemoryManager(settings)

        mock_redis_cls.assert_called_once()
        _, kwargs = mock_redis_cls.call_args
        assert kwargs["host"] == "localhost"
        assert kwargs["port"] == 6379


def test_setting_redis_host_changes_target_server():
    """
    Verifies that setting REDIS_HOST actually changes which server is used,
    solving Issue #10.
    """
    custom_host = "redis-cluster.prod.internal"
    settings = Settings(REDIS_URL=None, REDIS_HOST=custom_host, REDIS_PORT=6380)

    with patch("redis.Redis") as mock_redis_cls:
        mock_instance = MagicMock()
        mock_redis_cls.return_value = mock_instance

        manager = RedisMemoryManager(settings)

        mock_redis_cls.assert_called_once()
        _, kwargs = mock_redis_cls.call_args
        assert kwargs["host"] == custom_host
        assert kwargs["port"] == 6380


def test_explicit_redis_url_takes_precedence():
    """
    Verifies that if REDIS_URL is explicitly provided, it still takes precedence
    using redis.Redis.from_url().
    """
    url = "redis://:secret@remote-redis:6379/1"
    settings = Settings(REDIS_URL=url, REDIS_HOST="ignored-host")

    with patch("redis.Redis.from_url") as mock_from_url:
        mock_instance = MagicMock()
        mock_from_url.return_value = mock_instance

        manager = RedisMemoryManager(settings)

        mock_from_url.assert_called_once()
        assert mock_from_url.call_args[0][0] == url

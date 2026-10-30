"""
Unit tests for RAGService health check and /health endpoint integration.
Verifies that RAGService.health_check() accurately reflects real connection state
and does not falsely report connected=True just because the storage folder exists.
"""

from unittest.mock import MagicMock
from config import Settings
from rag import RAGService
from api import health


def test_rag_health_check_reports_false_when_disconnected():
    """
    Verifies that when _vector_store is None (uninitialized or failed to load),
    health_check() returns connected=False even if the storage directory exists.
    """
    rag = RAGService(Settings())
    rag._vector_store = None

    health_info = rag.health_check()
    assert health_info["connected"] is False
    # Folder exists on disk, but connected must strictly reflect the client status
    assert "storage_path_exists" in health_info


def test_rag_health_check_reports_true_when_connected():
    """
    Verifies that when _vector_store is active, health_check() returns connected=True.
    """
    rag = RAGService(Settings())
    rag._vector_store = MagicMock()

    health_info = rag.health_check()
    assert health_info["connected"] is True


def test_api_health_endpoint_reflects_truthful_rag_state():
    """
    Verifies that api.health() returns sensible status and accurate rag metadata.
    """
    resp = health()
    assert resp.status in ("healthy", "degraded")
    assert isinstance(resp.rag, dict)
    assert "connected" in resp.rag
    assert "storage_path_exists" in resp.rag


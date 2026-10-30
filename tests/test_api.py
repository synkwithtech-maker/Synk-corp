"""
Unit tests for FastAPI endpoints in api.py, specifically /chat error handling and status codes.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app
from constants import SYSTEM_ERROR_FALLBACK

client = TestClient(app, raise_server_exceptions=False)


def test_chat_returns_503_when_llm_service_fails():
    """
    Verifies that when the downstream LLM or ask_question_detailed raises an exception,
    the API returns HTTP 503 SERVICE UNAVAILABLE with SYSTEM_ERROR_FALLBACK detail.
    """
    with patch("api.ask_question_detailed", side_effect=RuntimeError("OpenRouter 503 service unavailable")):
        response = client.post(
            "/chat",
            json={"question": "What is Synk Corp?", "session_id": "test-session-503"},
        )
        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == SYSTEM_ERROR_FALLBACK


def test_chat_returns_200_on_success():
    """
    Verifies that when ask_question_detailed succeeds, /chat returns HTTP 200 with the answer.
    """
    mock_result = {
        "session_id": "test-session-200",
        "question": "What is Synk Corp?",
        "answer": "Synk Corp is a technology company.",
        "sources": ["overview.md"],
        "guardrail_passed": True,
        "guardrail_category": "clean",
    }
    with patch("api.ask_question_detailed", return_value=mock_result):
        response = client.post(
            "/chat",
            json={"question": "What is Synk Corp?", "session_id": "test-session-200"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Synk Corp is a technology company."
        assert data["sources"] == ["overview.md"]
        assert data["guardrail_passed"] is True
        assert data["guardrail_category"] == "clean"
        assert "latency_ms" in data


def test_chat_returns_400_on_empty_question():
    """
    Verifies that submitting an empty question returns HTTP 400 Bad Request.
    """
    response = client.post(
        "/chat",
        json={"question": "   ", "session_id": "test-session"},
    )
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]


def test_chat_returns_400_on_empty_session_id():
    """
    Verifies that submitting an empty session_id returns HTTP 400 Bad Request.
    """
    response = client.post(
        "/chat",
        json={"question": "Hello", "session_id": "   "},
    )
    assert response.status_code == 400
    assert "Session ID cannot be empty" in response.json()["detail"]

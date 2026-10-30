"""
Unit tests verifying application constants and cross-module synchronization.
Ensures that guardrails and chatbot share the single-source-of-truth constants.
"""

from constants import (
    NO_CONTEXT_REFUSAL,
    PROMPT_INJECTION_REFUSAL,
    SYSTEM_PROMPT_LEAK_FALLBACK,
    SYSTEM_ERROR_FALLBACK,
    DEFAULT_SAFETY_REFUSAL,
)
from guardrails import GuardrailsService
from config import Settings


def test_no_context_refusal_is_used_by_guardrails():
    """Verifies that GuardrailsService uses NO_CONTEXT_REFUSAL for strict grounding."""
    guard = GuardrailsService(Settings())
    res = guard.check_output(answer="Some ungrounded long answer" * 15, context="")
    assert res.category == "grounding_enforced"
    assert res.sanitized_text == NO_CONTEXT_REFUSAL


def test_constants_defined_and_non_empty():
    """Verifies all centralized response templates are non-empty strings."""
    for const in (
        NO_CONTEXT_REFUSAL,
        PROMPT_INJECTION_REFUSAL,
        SYSTEM_PROMPT_LEAK_FALLBACK,
        SYSTEM_ERROR_FALLBACK,
        DEFAULT_SAFETY_REFUSAL,
    ):
        assert isinstance(const, str)
        assert len(const.strip()) > 0

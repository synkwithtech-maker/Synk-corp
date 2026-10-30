"""
Unit tests for SupportChatbotEngine in chatbot.py.
Verifies that PII masked category is preserved through the output guardrail node
and that full workflow retains guardrail_category = "pii_masked".
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from config import Settings
from chatbot import SupportChatbotEngine


def test_output_guardrail_node_preserves_pii_masked():
    """
    Verifies that _output_guardrail_node preserves state['guardrail_category'] = 'pii_masked'
    when the generated answer is clean, rather than overwriting it with 'clean'.
    """
    engine = SupportChatbotEngine(settings=Settings())
    state = {
        "answer": "Here is the information about Synk Corp services.",
        "context": "Synk Corp provides software and consulting services.",
        "guardrail_category": "pii_masked",
    }
    result = engine._output_guardrail_node(state)
    assert result["guardrail_category"] == "pii_masked"


def test_output_guardrail_node_overwrites_on_actual_issue():
    """
    Verifies that _output_guardrail_node does overwrite the category
    when an actual output safety issue (such as system prompt leak) is detected.
    """
    engine = SupportChatbotEngine(settings=Settings())
    state = {
        "answer": "You are the customer support assistant for Synk Corp and follow rules.",
        "context": "Context",
        "guardrail_category": "pii_masked",
    }
    result = engine._output_guardrail_node(state)
    assert result["guardrail_category"] == "system_prompt_leak_prevented"


def test_question_with_card_number_returns_pii_masked():
    """
    End-to-end ask() test verifying that a question containing a credit card number
    returns guardrail_category = 'pii_masked' and redacts the card.
    """
    settings = Settings()
    engine = SupportChatbotEngine(settings=settings)

    # Mock RAG retrieval so it returns dummy context
    engine.rag_service.retrieve_context = MagicMock(return_value=[
        Document(page_content="Synk Corp support details.", metadata={"source": "test.txt"})
    ])

    # Mock LLM generation
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="We received your inquiry. How can we help?")
    engine._llm = mock_llm

    # Mock memory manager so Redis is not required
    engine.memory_manager.get_messages = MagicMock(return_value=[])
    engine.memory_manager.append_turn = MagicMock()

    question_with_card = "My card number is 4111 1111 1111 1111, can you check my billing?"
    response = engine.ask(question=question_with_card, session_id="test-session-123")

    assert response["guardrail_category"] == "pii_masked"
    assert response["guardrail_passed"] is True
    assert "[REDACTED_CREDIT_CARD]" in response["question"]
    assert "4111 1111 1111 1111" not in response["question"]



def test_llm_failure_propagates_and_does_not_save_to_memory():
    """
    Verifies that when LLM generation fails (e.g. OpenRouter outage),
    the exception propagates out of ask() and the failed turn is NOT saved to memory.
    """
    settings = Settings()
    engine = SupportChatbotEngine(settings=settings)

    # Mock RAG retrieval
    engine.rag_service.retrieve_context = MagicMock(return_value=[
        Document(page_content="Synk Corp support details.", metadata={"source": "test.txt"})
    ])

    # Mock LLM failure
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("OpenRouter service is down")
    engine._llm = mock_llm

    # Mock memory manager
    engine.memory_manager.get_messages = MagicMock(return_value=[])
    engine.memory_manager.append_turn = MagicMock()

    # Verify that the exception propagates
    with pytest.raises(RuntimeError, match="OpenRouter service is down"):
        engine.ask(question="What services does Synk Corp provide?", session_id="test-session-err")

    # Verify memory_manager.append_turn was NEVER called
    engine.memory_manager.append_turn.assert_not_called()

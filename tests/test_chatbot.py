"""
Unit tests for SupportChatbotEngine in chatbot.py.
Verifies that PII masked category is preserved through the output guardrail node
and that full workflow retains guardrail_category = "pii_masked".
"""

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

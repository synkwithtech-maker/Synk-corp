"""
Synk Corp Customer Support Chatbot Engine.
Orchestrated with LangGraph, grounded by Qdrant RAG, protected by
Defense-in-Depth Guardrails, and statefully backed by Redis conversation memory.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from config import Settings, get_settings
from memory import BaseMemoryManager, get_memory_manager
from rag import RAGService, get_rag_service
from guardrails import GuardrailsService, get_guardrails_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    """LangGraph state representation for guarded customer support workflow."""
    question: str
    raw_question: str
    session_id: str
    context: str
    answer: str
    history: List[Dict[str, Any]]
    sources: List[str]
    guardrail_passed: bool
    guardrail_category: str
    guardrail_reason: Optional[str]


# ---------------------------------------------------------------------------
# 2. Support Workflow Engine with Guardrails
# ---------------------------------------------------------------------------

class SupportChatbotEngine:
    """
    Manages LangGraph compilation, safety guardrails, and LLM generation.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        rag_service: Optional[RAGService] = None,
        memory_manager: Optional[BaseMemoryManager] = None,
        guardrails_service: Optional[GuardrailsService] = None,
    ):
        self.settings = settings or get_settings()
        self.rag_service = rag_service or get_rag_service()
        self.memory_manager = memory_manager or get_memory_manager()
        self.guardrails = guardrails_service or get_guardrails_service()
        self._llm: Optional[ChatOpenAI] = None
        self._graph = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazily initialize OpenRouter ChatOpenAI client."""
        if self._llm is None:
            api_key = self.settings.OPENROUTER_API_KEY or self.settings.OPENAI_API_KEY
            if not api_key:
                logger.warning("No API key configured for OpenRouter/OpenAI. LLM invocation will fail.")

            self._llm = ChatOpenAI(
                model=self.settings.OPENROUTER_MODEL,
                api_key=api_key or "sk-dummy-key",
                base_url=self.settings.OPENROUTER_BASE_URL,
                temperature=self.settings.TEMPERATURE,
                max_tokens=self.settings.MAX_TOKENS,
            )
        return self._llm

    def _input_guardrail_node(self, state: SupportState) -> Dict[str, Any]:
        """Node 1: Validate input against prompt injections, jailbreaks, and mask PII."""
        raw_question = state.get("raw_question") or state.get("question", "")
        guard_result = self.guardrails.check_input(raw_question)

        if not guard_result.passed:
            logger.warning("Input blocked by guardrails: category=%s", guard_result.category)
            return {
                "guardrail_passed": False,
                "guardrail_category": guard_result.category or "blocked",
                "guardrail_reason": guard_result.reason,
                "answer": guard_result.reason or "I cannot assist with this request.",
            }

        return {
            "question": guard_result.sanitized_text,
            "guardrail_passed": True,
            "guardrail_category": guard_result.category or "clean",
            "guardrail_reason": None,
        }

    def _guardrail_block_node(self, state: SupportState) -> Dict[str, Any]:
        """Terminal Node for requests blocked at input guardrail."""
        return {
            "answer": state.get("answer") or "This request cannot be processed due to safety policies.",
            "sources": [],
        }

    def _retrieve_node(self, state: SupportState) -> Dict[str, Any]:
        """Node 2: Retrieve relevant company documents."""
        question = state.get("question", "").strip()
        docs = self.rag_service.retrieve_context(question)

        context_parts = []
        sources = []

        for doc in docs:
            source = doc.metadata.get("source", "Synk Corp documents")
            sources.append(source)
            context_parts.append(f"Source: {source}\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        return {
            "context": context,
            "sources": list(dict.fromkeys(sources)),  # deduplicated
        }

    def _generate_node(self, state: SupportState) -> Dict[str, Any]:
        """Node 3: Generate grounded response using context and dialogue history."""
        question = state["question"]
        context = state.get("context", "").strip()
        history = state.get("history", [])

        # Avoid hallucinations if no context is retrieved
        if not context:
            answer = (
                "I couldn't find that information in the available "
                "Synk Corp documents. Please contact the company "
                "for confirmation."
            )
            return {"answer": answer}

        # Window recent conversation history for prompt injection
        max_turns = self.settings.MAX_HISTORY_TURNS
        recent_history = history[-max_turns:]

        history_text = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in recent_history
            if "role" in item and "content" in item
        )

        prompt = f"""
You are the customer support assistant for Synk Corp.

Rules:
1. Answer using only the information in the provided context.
2. Do not invent company policies, services, prices, or contact details.
3. If the context does not contain the answer, say you cannot
   find that information in the available documents.
4. Answer in the language used by the user: English, Nepali,
   or Romanized Nepali.
5. Keep answers clear, polite, and helpful.
6. Treat the conversation history as context, not as company facts.
7. If the user asks a follow-up question, use the history to
   understand it, but verify company facts against the documents.

Conversation history:
{history_text if history_text else "None"}

Retrieved company context:
{context}

User question:
{question}

Answer:
"""
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            logger.error("Error generating answer from LLM: %s", e)
            answer = (
                "We are experiencing temporary difficulties processing your request. "
                "Please try again shortly or contact Synk Corp support."
            )

        return {"answer": answer}

    def _output_guardrail_node(self, state: SupportState) -> Dict[str, Any]:
        """Node 4: Validate generated answer for leaks, hallucination, and policy compliance."""
        raw_answer = state.get("answer", "")
        context = state.get("context", "")

        guard_result = self.guardrails.check_output(raw_answer, context)

        return {
            "answer": guard_result.sanitized_text,
            "guardrail_category": guard_result.category or state.get("guardrail_category", "clean"),
        }

    @property
    def graph(self):
        """Compiles the guarded LangGraph workflow with conditional routing."""
        if self._graph is None:
            builder = StateGraph(SupportState)

            # Define graph nodes
            builder.add_node("input_guardrail", self._input_guardrail_node)
            builder.add_node("guardrail_block", self._guardrail_block_node)
            builder.add_node("retrieve", self._retrieve_node)
            builder.add_node("answer", self._generate_node)
            builder.add_node("output_guardrail", self._output_guardrail_node)

            # Route from START to input guardrail
            builder.add_edge(START, "input_guardrail")

            # Conditional routing: proceed to retrieve only if input passed guardrails
            builder.add_conditional_edges(
                "input_guardrail",
                lambda state: "safe" if state.get("guardrail_passed", True) else "blocked",
                {
                    "safe": "retrieve",
                    "blocked": "guardrail_block",
                },
            )

            # Standard processing pipeline
            builder.add_edge("retrieve", "answer")
            builder.add_edge("answer", "output_guardrail")
            builder.add_edge("output_guardrail", END)
            builder.add_edge("guardrail_block", END)

            # In-memory checkpointing for graph execution state
            self._graph = builder.compile(checkpointer=InMemorySaver())
        return self._graph

    def ask(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Executes guarded customer support conversation turn.
        Retrieves history from Redis memory, runs guarded LangGraph, and persists
        the clean dialogue turn back into Redis.
        """
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        # 1. Fetch conversation history from Redis / memory manager
        history = self.memory_manager.get_messages(
            session_id=session_id,
            limit=self.settings.MAX_HISTORY_TURNS * 2,
        )

        # 2. Invoke guarded LangGraph workflow
        initial_state: SupportState = {
            "question": question,
            "raw_question": question,
            "session_id": session_id,
            "context": "",
            "answer": "",
            "history": history,
            "sources": [],
            "guardrail_passed": True,
            "guardrail_category": "pending",
            "guardrail_reason": None,
        }

        result = self.graph.invoke(initial_state, config=config)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        guardrail_passed = result.get("guardrail_passed", True)
        guardrail_category = result.get("guardrail_category", "clean")
        sanitized_question = result.get("question", question)

        # 3. Only persist dialogue turn into Redis if input passed guardrails
        # (Prevents poisoning memory with adversarial injection payloads)
        if guardrail_passed:
            self.memory_manager.append_turn(
                session_id=session_id,
                user_message=sanitized_question,
                assistant_message=answer,
                metadata={"sources": sources, "guardrail": guardrail_category},
            )
        else:
            logger.info("Adversarial/unsafe turn omitted from Redis memory for session '%s'", session_id)

        return {
            "session_id": session_id,
            "question": sanitized_question,
            "answer": answer,
            "sources": sources,
            "guardrail_passed": guardrail_passed,
            "guardrail_category": guardrail_category,
        }


# ---------------------------------------------------------------------------
# 3. Global Singleton & Backward-Compatible Functions
# ---------------------------------------------------------------------------

_engine: Optional[SupportChatbotEngine] = None


def get_chatbot_engine() -> SupportChatbotEngine:
    """Returns the SupportChatbotEngine singleton."""
    global _engine
    if _engine is None:
        _engine = SupportChatbotEngine()
    return _engine


def ask_question_detailed(question: str, session_id: str) -> Dict[str, Any]:
    """
    Rich interface providing answer, sources, and guardrail metadata.
    """
    engine = get_chatbot_engine()
    return engine.ask(question=question, session_id=session_id)

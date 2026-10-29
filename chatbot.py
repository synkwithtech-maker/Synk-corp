"""
Synk Corp Customer Support Chatbot Engine.
Orchestrated with LangGraph, grounded by Qdrant RAG, protected by
Defense-in-Depth Guardrails, and statefully backed by Redis conversation memory.
Supports both synchronous completion and real-time Server-Sent Events (SSE) token streaming.
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

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
# 2. Support Workflow Engine with Guardrails & Streaming
# ---------------------------------------------------------------------------

class SupportChatbotEngine:
    """
    Manages LangGraph compilation, safety guardrails, token streaming, and LLM generation.
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
        """Lazily initialize OpenRouter ChatOpenAI client with streaming enabled."""
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
                streaming=True,
            )
        return self._llm

    def _build_prompt(
        self,
        question: str,
        context: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """Constructs the system prompt with grounding rules, history, and context."""
        max_turns = self.settings.MAX_HISTORY_TURNS
        recent_history = history[-max_turns:]

        history_text = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in recent_history
            if "role" in item and "content" in item
        )

        return f"""You are the customer support assistant for Synk Corp.

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

        prompt = self._build_prompt(
            question=question,
            context=context,
            history=history,
        )

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
        Synchronous customer support turn.
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

    async def ask_stream(
        self, question: str, session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Real-time token streaming generator for customer support queries.
        Emits events: start, guardrail_blocked, sources, token, done.
        """
        start_time = time.perf_counter()

        yield {
            "event": "start",
            "session_id": session_id,
        }

        # 1. Input Guardrail
        raw_question = question.strip()
        guard_result = self.guardrails.check_input(raw_question)

        if not guard_result.passed:
            logger.warning("Streaming input blocked by guardrails: category=%s", guard_result.category)
            refusal = guard_result.reason or "I cannot assist with this request."
            yield {
                "event": "guardrail_blocked",
                "category": guard_result.category or "blocked",
                "answer": refusal,
            }
            yield {
                "event": "token",
                "token": refusal,
            }
            yield {
                "event": "done",
                "session_id": session_id,
                "answer": refusal,
                "sources": [],
                "guardrail_passed": False,
                "guardrail_category": guard_result.category or "blocked",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }
            return

        sanitized_question = guard_result.sanitized_text

        # 2. Retrieve Context (in async worker thread to keep event loop free)
        docs = await asyncio.to_thread(self.rag_service.retrieve_context, sanitized_question)
        context_parts = []
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Synk Corp documents")
            sources.append(source)
            context_parts.append(f"Source: {source}\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        sources = list(dict.fromkeys(sources))

        yield {
            "event": "sources",
            "sources": sources,
        }

        # 3. Check for empty context
        if not context.strip():
            fallback_answer = (
                "I couldn't find that information in the available "
                "Synk Corp documents. Please contact the company "
                "for confirmation."
            )
            yield {
                "event": "token",
                "token": fallback_answer,
            }
            await asyncio.to_thread(
                self.memory_manager.append_turn,
                session_id,
                sanitized_question,
                fallback_answer,
                {"sources": []},
            )
            yield {
                "event": "done",
                "session_id": session_id,
                "answer": fallback_answer,
                "sources": [],
                "guardrail_passed": True,
                "guardrail_category": "clean",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }
            return

        # 4. Fetch recent dialogue history from Redis
        history = await asyncio.to_thread(
            self.memory_manager.get_messages,
            session_id,
            self.settings.MAX_HISTORY_TURNS * 2,
        )

        prompt = self._build_prompt(
            question=sanitized_question,
            context=context,
            history=history,
        )

        # 5. Stream LLM tokens chunk-by-chunk
        full_answer_parts = []
        try:
            async for chunk in self.llm.astream(prompt):
                content = chunk.content
                if isinstance(content, str) and content:
                    full_answer_parts.append(content)
                    yield {
                        "event": "token",
                        "token": content,
                    }
        except Exception as e:
            logger.error("Error during streaming generation: %s", e)
            error_msg = "\n[Error: Stream interrupted. Please try again.]"
            full_answer_parts.append(error_msg)
            yield {
                "event": "token",
                "token": error_msg,
            }

        full_answer = "".join(full_answer_parts)

        # 6. Output Guardrail Check
        output_guard = self.guardrails.check_output(full_answer, context)
        final_answer = output_guard.sanitized_text
        if final_answer != full_answer:
            yield {
                "event": "guardrail_adjusted",
                "answer": final_answer,
            }

        # 7. Persist complete dialogue turn to Redis
        await asyncio.to_thread(
            self.memory_manager.append_turn,
            session_id,
            sanitized_question,
            final_answer,
            {"sources": sources, "streaming": True},
        )

        # 8. Done event with timing and sources
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        yield {
            "event": "done",
            "session_id": session_id,
            "answer": final_answer,
            "sources": sources,
            "guardrail_passed": True,
            "guardrail_category": output_guard.category or "clean",
            "latency_ms": latency_ms,
        }


# ---------------------------------------------------------------------------
# 3. Global Singleton & Exported Functions
# ---------------------------------------------------------------------------

_engine: Optional[SupportChatbotEngine] = None


def get_chatbot_engine() -> SupportChatbotEngine:
    """Returns the SupportChatbotEngine singleton."""
    global _engine
    if _engine is None:
        _engine = SupportChatbotEngine()
    return _engine


def ask_question(question: str, session_id: str) -> str:
    """
    Standard interface invoked by API endpoints.
    Maintains 100% backward compatibility by returning the string answer.
    """
    engine = get_chatbot_engine()
    result = engine.ask(question=question, session_id=session_id)
    return result["answer"]


def ask_question_detailed(question: str, session_id: str) -> Dict[str, Any]:
    """
    Rich interface providing answer, sources, and guardrail metadata.
    """
    engine = get_chatbot_engine()
    return engine.ask(question=question, session_id=session_id)


async def stream_chat_events(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Formats the ask_stream generator into standard SSE (Server-Sent Events) chunks.
    Format:
        event: <event_name>\n
        data: <json_string>\n\n
    Final message:
        data: [DONE]\n\n
    """
    engine = get_chatbot_engine()
    async for item in engine.ask_stream(question=question, session_id=session_id):
        event_name = item.get("event", "message")
        payload = json.dumps(item)
        yield f"event: {event_name}\ndata: {payload}\n\n"
    yield "data: [DONE]\n\n"

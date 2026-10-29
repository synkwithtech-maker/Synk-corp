"""
Chatbot request and response schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's question or message",
        examples=["What services does Synk Corp provide?"],
    )
    session_id: str = Field(
        default="default-session",
        min_length=1,
        max_length=128,
        description="Unique session identifier for Redis conversation tracking",
        examples=["customer-session-123"],
    )


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list, description="Documents referenced for the answer")
    guardrail_passed: bool = Field(default=True, description="Whether query passed safety guardrails")
    guardrail_category: str = Field(default="clean", description="Safety category: clean, pii_masked, prompt_injection, system_leak_prevented")
    latency_ms: float = Field(..., description="Request processing time in milliseconds")


class DialogueMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    history: List[DialogueMessage]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool
    message: str

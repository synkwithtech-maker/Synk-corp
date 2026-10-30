"""
Production-grade Guardrails Module for Synk Corp Support Chatbot.
Provides defense-in-depth security:
1. Input Guardrails: Prompt injection detection, jailbreak prevention, PII masking.
2. Output Guardrails: System prompt leak prevention, strict context grounding verification.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

from config import Settings, get_settings
from constants import NO_CONTEXT_REFUSAL, PROMPT_INJECTION_REFUSAL, SYSTEM_PROMPT_LEAK_FALLBACK

logger = logging.getLogger("synkcorp.guardrails")


@dataclass
class GuardrailResult:
    """Represents the outcome of a guardrail evaluation."""
    passed: bool
    sanitized_text: str
    category: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputGuardrail:
    """
    Validates and sanitizes incoming user queries before they reach
    the vector database or language model.
    """

    # High-confidence prompt injection and jailbreak patterns
    INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts|directions)", re.IGNORECASE),
        re.compile(r"(?:disregard|forget)\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules|prompts)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:dan|an\s+unrestricted\s+ai|jailbroken|in\s+developer\s+mode)", re.IGNORECASE),
        re.compile(r"(?:repeat|reveal|print|show|output|leak)\s+(?:the\s+)?(?:system\s+prompt|developer\s+instructions|secret\s+instructions)", re.IGNORECASE),
        re.compile(r"what\s+are\s+your\s+(?:exact\s+)?(?:system\s+instructions|system\s+prompts|developer\s+rules)", re.IGNORECASE),
        re.compile(r"(?:bypass|disable|override)\s+(?:content\s+filter|guardrails|safety\s+protocols|all\s+rules)", re.IGNORECASE),
        re.compile(r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[SYSTEM\])", re.IGNORECASE),
        re.compile(r"(?:pretend|act\s+as\s+if)\s+you\s+have\s+no\s+(?:rules|filters|morals|limits)", re.IGNORECASE),
    ]

    # Personally Identifiable Information (PII) redaction regexes
    CREDIT_CARD_PATTERN: Pattern = re.compile(
        r"\b(?:\d{4}[ -]?){3}\d{4}\b"
    )
    SSN_PATTERN: Pattern = re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    )
    API_KEY_PATTERN: Pattern = re.compile(
        r"\b(?:sk-[a-zA-Z0-9_\-]{20,}|ghp_[a-zA-Z0-9]{20,}|bearer\s+[a-zA-Z0-9_\-\.]{25,})\b",
        re.IGNORECASE,
    )

    def __init__(self, settings: Settings):
        self.settings = settings

    def mask_pii(self, text: str) -> tuple[str, bool]:
        """Redacts sensitive PII (credit cards, SSNs, secret keys) from user input."""
        masked = text
        detected = False

        if self.CREDIT_CARD_PATTERN.search(masked):
            masked = self.CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", masked)
            detected = True

        if self.SSN_PATTERN.search(masked):
            masked = self.SSN_PATTERN.sub("[REDACTED_SSN]", masked)
            detected = True

        if self.API_KEY_PATTERN.search(masked):
            masked = self.API_KEY_PATTERN.sub("[REDACTED_SECRET_KEY]", masked)
            detected = True

        return masked, detected

    def detect_prompt_injection(self, text: str) -> Optional[str]:
        """Scans for adversarial prompt injections, jailbreaks, and rule bypasses."""
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_phrase = match.group(0)
                logger.warning("Prompt injection detected: '%s'", matched_phrase)
                return matched_phrase
        return None

    def validate(self, text: str) -> GuardrailResult:
        """Executes complete input guardrail pipeline."""
        if not self.settings.GUARDRAILS_ENABLED:
            return GuardrailResult(passed=True, sanitized_text=text)

        cleaned_text = text.strip()

        # 1. Prompt Injection & Jailbreak check
        if self.settings.GUARDRAIL_BLOCK_PROMPT_INJECTION:
            injection_match = self.detect_prompt_injection(cleaned_text)
            if injection_match:
                return GuardrailResult(
                    passed=False,
                    sanitized_text="",
                    category="prompt_injection",
                    reason=PROMPT_INJECTION_REFUSAL,
                    metadata={"matched_pattern": injection_match},
                )

        # 2. PII Detection and Masking
        sanitized = cleaned_text
        pii_found = False
        if self.settings.GUARDRAIL_MASK_PII:
            sanitized, pii_found = self.mask_pii(cleaned_text)

        return GuardrailResult(
            passed=True,
            sanitized_text=sanitized,
            category="pii_masked" if pii_found else "clean",
            metadata={"pii_detected": pii_found},
        )


class OutputGuardrail:
    """
    Validates model generation before returning it to the client to guarantee
    grounding, prevent system prompt leakage, and maintain safety.
    """

    SYSTEM_LEAK_PHRASES: List[str] = [
        "you are the customer support assistant for synk corp",
        "rules: 1. answer using only the information",
        "retrieved company context:",
        "treat the conversation history as context, not as company facts",
        "<|im_start|>",
        "<|im_end|>",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings

    def detect_system_leak(self, text: str) -> bool:
        """Ensures the generated text doesn't contain raw system prompt fragments."""
        lower_text = text.lower()
        for phrase in self.SYSTEM_LEAK_PHRASES:
            if phrase in lower_text:
                logger.warning("System instruction leak detected in output: '%s'", phrase)
                return True
        return False

    def validate(self, answer: str, context: str) -> GuardrailResult:
        """Executes complete output guardrail pipeline."""
        if not self.settings.GUARDRAILS_ENABLED:
            return GuardrailResult(passed=True, sanitized_text=answer)

        # 1. System Prompt Leak Check
        if self.detect_system_leak(answer):
            return GuardrailResult(
                passed=False,
                sanitized_text=SYSTEM_PROMPT_LEAK_FALLBACK,
                category="system_prompt_leak_prevented",
                reason="System prompt leak was detected and suppressed.",
            )

        # 2. Strict Grounding Validation
        # If no context was found, enforce standard refusal instead of hallucinating facts
        if self.settings.GUARDRAIL_STRICT_GROUNDING and not context.strip():
            safe_refusal = NO_CONTEXT_REFUSAL
            # If the model tried to answer anyway without context, sanitize to safe refusal
            if answer.strip() != safe_refusal and len(answer) > 200:
                logger.info("Enforcing strict grounding fallback for ungrounded answer.")
                return GuardrailResult(
                    passed=True,
                    sanitized_text=safe_refusal,
                    category="grounding_enforced",
                    reason="Answer adjusted to maintain strict adherence to corporate documents.",
                )

        return GuardrailResult(
            passed=True,
            sanitized_text=answer,
            category="clean",
        )


class GuardrailsService:
    """Enterprise composite guardrails service."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.input_guard = InputGuardrail(self.settings)
        self.output_guard = OutputGuardrail(self.settings)

    def check_input(self, text: str) -> GuardrailResult:
        return self.input_guard.validate(text)

    def check_output(self, answer: str, context: str) -> GuardrailResult:
        return self.output_guard.validate(answer, context)


# Singleton instance
_guardrails_service: Optional[GuardrailsService] = None


def get_guardrails_service() -> GuardrailsService:
    """Returns the GuardrailsService singleton."""
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service

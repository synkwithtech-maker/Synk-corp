"""
Application-wide constants for Synk Corp Support Chatbot.
Centralizes user-facing messages, guardrail refusals, and fallback templates
so messages are defined in exactly one place.
"""

# Standard refusal when no corporate knowledge context is found
NO_CONTEXT_REFUSAL = (
    "I couldn't find that information in the available "
    "Synk Corp documents. Please contact the company "
    "for confirmation."
)

# Refusal when prompt injection or system override is detected
PROMPT_INJECTION_REFUSAL = (
    "I am designed solely to assist with Synk Corp customer inquiries. "
    "I cannot execute system override commands."
)

# Safe replacement greeting when system prompt leakage is suppressed
SYSTEM_PROMPT_LEAK_FALLBACK = (
    "I am the customer support assistant for Synk Corp. "
    "How can I assist you with our services today?"
)

# Fallback message during unexpected LLM processing errors
SYSTEM_ERROR_FALLBACK = (
    "We are experiencing temporary difficulties processing your request. "
    "Please try again shortly or contact Synk Corp support."
)

# Default terminal refusal for blocked requests
DEFAULT_SAFETY_REFUSAL = "This request cannot be processed due to safety policies."

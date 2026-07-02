"""
Guardrails module — defense-in-depth pre-filter for prompt injection and off-topic requests.

This runs BEFORE the LLM call. It catches known injection patterns and obvious
off-topic requests via regex/keyword matching. The system prompt handles the
semantic layer (e.g. "should I give my new hire a raise").
"""

import re
from typing import Optional


# Compiled regex patterns for known prompt-injection phrasing.
# Each pattern is case-insensitive and matches common variants.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bignore\s+(all\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?instructions?\b",
        r"\bdisregard\s+(all\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?instructions?\b",
        r"\bforget\s+(all\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?instructions?\b",
        r"\byou\s+are\s+now\b",
        r"\bact\s+as\s+(a|an)?\s*(unrestricted|unfiltered|uncensored|evil|malicious)\b",
        r"\bpretend\s+(you\s+are|to\s+be)\b",
        r"\breveal\s+(your\s+)?(system\s+)?prompt\b",
        r"\bshow\s+(me\s+)?(your\s+)?(system\s+)?prompt\b",
        r"\bprint\s+(your\s+)?(system\s+)?prompt\b",
        r"\bdisplay\s+(your\s+)?(system\s+)?(prompt|instructions?)\b",
        r"\boutput\s+(your\s+)?(system\s+)?(prompt|instructions?)\b",
        r"\bwhat\s+(is|are)\s+your\s+(system\s+)?(instructions?|prompt)\b",
        r"\brepeat\s+(your\s+)?(system\s+)?(prompt|instructions?)\b",
        r"\bsystem\s+prompt\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        r"\bDAN\b",  # "Do Anything Now"
        r"\bdo\s+anything\s+now\b",
        r"\bdeveloper\s+mode\b",
        r"\boverride\s+(your\s+)?(safety|content|instructions?)\b",
        r"\bbypass\s+(your\s+)?(safety|content|filter|restrictions?)\b",
        r"\brole\s*play\s+as\b",
    ]
]

# Canned refusal responses for different trigger types
_INJECTION_REFUSAL = (
    "I'm designed to help you select SHL assessments for your hiring needs. "
    "I can't process that kind of request. "
    "How can I help you find the right assessment solution?"
)

_OFF_TOPIC_REFUSAL = (
    "I specialize in recommending SHL assessments and can only help with "
    "assessment selection, comparison, and related questions. "
    "Could you tell me about the role you're hiring for so I can suggest "
    "relevant assessments?"
)


def check_injection(text: str) -> bool:
    """
    Returns True if the text matches any known prompt-injection pattern.
    This is a fast pre-filter — the system prompt provides the semantic layer.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def get_guardrail_refusal(messages: list) -> Optional[str]:
    """
    Check the latest user message for injection attempts.

    Args:
        messages: list of Message objects (or dicts with 'role' and 'content')

    Returns:
        A refusal string if an injection/off-topic trigger was found,
        None otherwise (meaning the message should proceed to the LLM).
    """
    if not messages:
        return None

    # Find the latest user message
    latest_user_msg = None
    for msg in reversed(messages):
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        if role == "user":
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            latest_user_msg = content
            break

    if latest_user_msg is None:
        return None

    # Check for injection patterns
    if check_injection(latest_user_msg):
        return _INJECTION_REFUSAL

    return None

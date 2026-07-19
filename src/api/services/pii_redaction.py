"""PII redaction layer (NFR-Privacy / US-209).

Applied as a mandatory gate immediately before any LLM call (see
services/explanation.py). No raw PII may leave the trust boundary: SSN,
DOB-shaped dates, and account-number-shaped digit sequences are redacted or,
if redaction still leaves a match, the call is blocked entirely rather than
risking a leak.
"""

import logging
import re
from typing import Tuple

logger = logging.getLogger("halcyon.pii_redaction")

_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "dob_slash": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "dob_iso": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "long_digit_sequence": re.compile(r"\b\d{8,}\b"),  # account/routing-number-shaped
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}


def redact_pii(text: str) -> str:
    """Replace any PII-shaped substring with a labelled redaction marker."""
    if not text:
        return text
    redacted = text
    for label, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted


def contains_pii(text: str) -> bool:
    """The linter: True if any PII-shaped pattern is still present."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _PATTERNS.values())


def sanitize_for_llm(text: str) -> Tuple[str, bool]:
    """Redact PII from text bound for an LLM prompt.

    Returns (sanitized_text, was_blocked). was_blocked=True means PII
    survived redaction (should not happen given the patterns above, but is
    the documented fail-closed path) - the caller must not send the text and
    must log the attempted leak (US-209 AC: "the call is blocked and logged").
    """
    sanitized = redact_pii(text)
    if contains_pii(sanitized):
        logger.warning("PII redaction: attempted leak blocked, prompt not sent to LLM")
        return sanitized, True
    return sanitized, False

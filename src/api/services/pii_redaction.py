"""PII redaction layer for LLM prompts (US-209 / NFR-Privacy).

Every prompt that would leave the trust boundary for an LLM call must pass
through :func:`sanitize_for_llm`, which (1) redacts known PII patterns and
(2) re-lints the redacted text; if any PII pattern still matches after
redaction, the call is *blocked* by raising :class:`PIILeakError` and the
event is logged, so no raw PII can leak even if a new field slips through.

This module has no LLM dependency of its own - it is the gate that sits in
front of whichever provider is chosen later (FR-9). It is safe to build and
enforce now regardless of that decision.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("halcyon.pii")

# Ordered patterns: (name, compiled regex, replacement token). Order matters -
# more specific patterns (SSN, card) run before looser ones (long digit runs).
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CARD]"),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[REDACTED_IBAN]"),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Date of birth: ISO or common slash/dash formats (run before bare-digit rules).
    ("DOB", re.compile(r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b"), "[REDACTED_DOB]"),
    ("DOB", re.compile(r"\b(?:0[1-9]|[12]\d|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b"), "[REDACTED_DOB]"),
    # Bank/account numbers: standalone 8-17 digit runs. A bare digit run is more
    # likely an account than a phone, so this runs before the (separator-based) phone rule.
    ("ACCOUNT_NUMBER", re.compile(r"\b\d{8,17}\b"), "[REDACTED_ACCOUNT]"),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"), "[REDACTED_PHONE]"),
]


class PIILeakError(Exception):
    """Raised when PII survives redaction - the LLM call must be blocked."""


def find_pii(text: str) -> list[dict]:
    """Return a list of {type, match} for every PII pattern found in text."""
    findings = []
    for name, pattern, _ in _PATTERNS:
        for m in pattern.finditer(text):
            findings.append({"type": name, "match": m.group(0)})
    return findings


def redact(text: str) -> tuple[str, list[dict]]:
    """Redact all known PII patterns. Returns (redacted_text, findings)."""
    if not text:
        return text, []
    findings = []
    redacted = text
    for name, pattern, token in _PATTERNS:
        matches = list(pattern.finditer(redacted))
        if matches:
            findings.extend({"type": name, "match": m.group(0)} for m in matches)
            redacted = pattern.sub(token, redacted)
    return redacted, findings


def sanitize_for_llm(text: str, *, context: str = "llm_call") -> str:
    """Redact PII and hard-verify the result before it leaves the trust boundary.

    Returns the redacted, PII-free prompt. Raises :class:`PIILeakError` (and
    logs) if any PII pattern still matches after redaction - the caller must
    treat that as a blocked call, never send the raw text.
    """
    redacted, findings = redact(text)
    if findings:
        logger.info(
            "pii_redacted context=%s counts=%s", context, _counts_by_type(findings)
        )
    residual = find_pii(redacted)
    if residual:
        logger.error(
            "pii_leak_blocked context=%s residual_types=%s", context, _counts_by_type(residual)
        )
        raise PIILeakError(
            f"PII leak blocked in {context}: {sorted({r['type'] for r in residual})}"
        )
    return redacted


def _counts_by_type(findings: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
    return counts

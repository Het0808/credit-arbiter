"""LLM-generated grounded explanation via OpenRouter (FR-9).

Turns the deterministic, source-tagged evidence *claims* into a fluent narrative
using an LLM through OpenRouter's OpenAI-compatible API. Safety rails:

  - The LLM only ever writes the *explanation*; the Approve/Refer/Decline decision
    stays deterministic (policy + model + evidence). The PRD forbids LLM auto-decisions.
  - The prompt is built solely from the grounded claims (each carrying its source),
    and is passed through the PII redaction gate before leaving the trust boundary.
  - Any failure (no key, network/provider error) falls back to the deterministic
    narrative, so explanations never hard-fail.

Config (env): LLM_PROVIDER=openrouter to enable; OPENROUTER_API_KEY (required),
OPENROUTER_MODEL (default openai/gpt-4o-mini), OPENROUTER_BASE_URL.
"""

from __future__ import annotations

import logging
import os

from .pii_redaction import sanitize_for_llm

logger = logging.getLogger("halcyon.llm")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_SYSTEM_PROMPT = (
    "You are a lending underwriting assistant. Write a concise, factual explanation "
    "(2-4 sentences) of the lending recommendation using ONLY the evidence bullet points "
    "provided. Cite policy clause IDs in square brackets, e.g. [POL-PL-001]. Do NOT invent "
    "any facts, numbers, names, or clauses beyond the evidence. Do NOT change or second-guess "
    "the recommendation; only explain it."
)


def llm_enabled() -> bool:
    return os.getenv("LLM_PROVIDER", "none").lower() == "openrouter" and bool(os.getenv("OPENROUTER_API_KEY"))


def generate_llm_narrative(claims: list[dict], recommendation: str) -> str:
    """Return an LLM-written narrative grounded in ``claims``. Raises on failure."""
    from openai import OpenAI

    evidence = "\n".join(f"- {c['text']} (source: {c['source']})" for c in claims)
    user_prompt = sanitize_for_llm(
        f"Recommendation: {recommendation}\n\nEvidence:\n{evidence}\n\n"
        "Write the underwriter-facing explanation.",
        context="llm_explanation",
    )

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def maybe_llm_narrative(claims: list[dict], recommendation: str) -> tuple[str | None, str]:
    """Return (narrative, generator). generator is 'llm' on success, else 'deterministic'
    (caller keeps the deterministic narrative)."""
    if not llm_enabled():
        return None, "deterministic"
    try:
        return generate_llm_narrative(claims, recommendation), "llm"
    except Exception as exc:
        logger.warning("LLM explanation failed, using deterministic narrative: %s", exc)
        return None, "deterministic"

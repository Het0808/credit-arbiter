"""Grounded explanation generation (FR-9 / US-208) via the Groq API.

Every factual claim must be traceable to a source (NFR-Explainability): the
prompt only ever contains aggregated evidence (risk band/score, SHAP factor
labels, retrieved policy clause IDs/text, regulatory + fairness results) -
never raw applicant PII - and is passed through the PII redaction linter
(services/pii_redaction.py, US-209) as a defense-in-depth check immediately
before the call. After generation, every policy clause ID the model cites is
checked against the clauses actually retrieved for this assessment; if the
model cites anything else, the response is discarded (a pragmatic stand-in
for the full hallucination-eval harness in US-404, not a claim of full
LLM-judge coverage) and a deterministic templated narrative is used instead.

No LLM call is ever a kill-switch trigger: if Groq is unavailable, times out,
or the API key is unset, generation silently degrades to the template - the
narrative is an enrichment of the evidence chain, not one of the AC-6 blocking
evidence components (risk, policy, regulatory, documents).
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("halcyon.explanation")

from .. import audit_log
from .pii_redaction import sanitize_for_llm

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Published Groq per-token pricing for the configured model (USD per 1M
# tokens), used only for the cost guardrail (US-402/AC-8) - not billed
# anywhere else in this project.
_INPUT_COST_PER_1M = 0.05
_OUTPUT_COST_PER_1M = 0.08
_MAX_OUTPUT_TOKENS = 300

# NFR cost ceiling: guardrail alarm above $0.08/application degrades to the
# human path automatically. Estimated pre-call (worst case: max_tokens all
# used) so a call that would breach the budget is never even sent.
COST_GUARDRAIL_USD = 0.08

_CLAUSE_ID_PATTERN = re.compile(r"\bPOL-[A-Z]{2}-\d{3}\b")

_client = None
_client_checked = False


def _get_client():
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq

        _client = Groq(api_key=api_key)
    except Exception as e:
        logger.warning(f"Groq client unavailable: {e}")
        _client = None
    return _client


def _build_prompt(evidence: Dict[str, Any], recommendation: str) -> str:
    lines = [
        f"Recommendation: {recommendation}",
        f"Risk band: {evidence.get('risk_band')} (probability of default: {evidence.get('risk_score')})",
    ]
    factors = evidence.get("risk_factors") or []
    if factors:
        lines.append("Top risk factors (SHAP):")
        for f in factors[:5]:
            direction = "increases" if f.get("impact", 0) >= 0 else "decreases"
            lines.append(f"  - {f.get('label')} {direction} default risk")

    clauses = evidence.get("retrieved_clauses") or []
    if clauses:
        lines.append(f"Applicable loan scheme: {evidence.get('loan_scheme')}")
        lines.append("Retrieved policy clauses:")
        for c in clauses:
            lines.append(f"  - [{c['clause_id']}] {c['title']}: {c['text']}")

    failed_rules = evidence.get("policy_failed_rules") or []
    if failed_rules:
        lines.append("Failed policy rules:")
        for r in failed_rules:
            lines.append(f"  - {r['rule']} ({r['clause_id']}): {r['detail']}")

    lines.append(f"Regulatory status: {evidence.get('regulatory_status')}")
    if evidence.get("thin_file_flag"):
        lines.append("Applicant is flagged thin-file (limited credit history).")

    return "\n".join(lines)


def _template_fallback(evidence: Dict[str, Any], recommendation: str) -> str:
    """Deterministic, zero-cost, zero-hallucination-risk narrative built
    directly from the evidence chain - always available regardless of LLM
    availability."""
    band = evidence.get("risk_band") or "an indeterminate"
    score = evidence.get("risk_score")
    score_text = f"{score * 100:.1f}%" if isinstance(score, (int, float)) else "unavailable"
    clauses = evidence.get("retrieved_clauses") or []
    clause_text = (
        "; ".join(f"{c['clause_id']} ({c['title']})" for c in clauses)
        if clauses
        else "no policy clause"
    )
    failed = evidence.get("policy_failed_rules") or []
    failed_text = f" Policy check failed: {len(failed)} rule(s) not met." if failed else ""
    thin_file_text = " Applicant is flagged thin-file." if evidence.get("thin_file_flag") else ""

    return (
        f"Recommendation: {recommendation}. Risk band {band} "
        f"({score_text} probability of default) under the {evidence.get('loan_scheme')} scheme, "
        f"grounded in {clause_text}. Regulatory status: {evidence.get('regulatory_status')}."
        f"{failed_text}{thin_file_text}"
    )


def generate_explanation(evidence: Dict[str, Any], recommendation: str) -> Dict[str, Any]:
    """Generate a grounded narrative explanation for one assessment.

    Returns {narrative: str, source: "llm"|"template", cost_usd: float,
    blocked_reason: str | None}.
    """
    valid_clause_ids = {c["clause_id"] for c in (evidence.get("retrieved_clauses") or [])}

    client = _get_client()
    if client is None:
        return {
            "narrative": _template_fallback(evidence, recommendation),
            "source": "template",
            "cost_usd": 0.0,
            "blocked_reason": "llm_not_configured",
        }

    prompt = _build_prompt(evidence, recommendation)
    sanitized_prompt, blocked = sanitize_for_llm(prompt)
    if blocked:
        return {
            "narrative": _template_fallback(evidence, recommendation),
            "source": "template",
            "cost_usd": 0.0,
            "blocked_reason": "pii_detected",
        }

    # Pre-call worst-case cost estimate (rough token count: 1 token ~= 4
    # chars). If even the worst case (max_tokens all consumed) would breach
    # the guardrail, skip the call entirely rather than risk it.
    estimated_prompt_tokens = len(sanitized_prompt) / 4
    worst_case_cost = (
        (estimated_prompt_tokens / 1_000_000) * _INPUT_COST_PER_1M
        + (_MAX_OUTPUT_TOKENS / 1_000_000) * _OUTPUT_COST_PER_1M
    )
    if worst_case_cost > COST_GUARDRAIL_USD:
        logger.warning(f"Estimated cost ${worst_case_cost:.4f} exceeds guardrail ${COST_GUARDRAIL_USD}, using template")
        return {
            "narrative": _template_fallback(evidence, recommendation),
            "source": "template",
            "cost_usd": 0.0,
            "blocked_reason": "cost_guardrail",
        }

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an underwriting copilot writing a short (3-5 sentence) "
                        "rationale for a human underwriter. Only use the facts given below. "
                        "Cite policy clause IDs exactly as given (e.g. POL-PL-001) when referencing "
                        "a policy rule. Never invent a clause ID, number, or fact not present in the evidence."
                    ),
                },
                {"role": "user", "content": sanitized_prompt},
            ],
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.2,
            timeout=8,
        )
        narrative = (response.choices[0].message.content or "").strip()
        usage = response.usage
        cost_usd = (
            (usage.prompt_tokens / 1_000_000) * _INPUT_COST_PER_1M
            + (usage.completion_tokens / 1_000_000) * _OUTPUT_COST_PER_1M
        )
        audit_log.log_external_call("groq", model=GROQ_MODEL, success=True, cost_usd=round(cost_usd, 6))

        cited_ids = set(_CLAUSE_ID_PATTERN.findall(narrative))
        hallucinated = cited_ids - valid_clause_ids
        if hallucinated:
            logger.warning(f"Explanation cited ungrounded clause IDs {hallucinated}, using template fallback")
            return {
                "narrative": _template_fallback(evidence, recommendation),
                "source": "template",
                "cost_usd": round(cost_usd, 6),
                "blocked_reason": "ungrounded_citation",
            }

        return {"narrative": narrative, "source": "llm", "cost_usd": round(cost_usd, 6), "blocked_reason": None}
    except Exception as e:
        logger.warning(f"Groq call failed, using template fallback: {e}")
        audit_log.log_external_call("groq", model=GROQ_MODEL, success=False, error=str(e))
        return {
            "narrative": _template_fallback(evidence, recommendation),
            "source": "template",
            "cost_usd": 0.0,
            "blocked_reason": "llm_call_failed",
        }

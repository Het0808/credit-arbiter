"""Per-application cost metering with hard cutoff (NFR-Cost / US-402).

Estimates the marginal cost of one assessment from the work actually done -
retrieval, model inference, regulatory calls, document checks, and the
projected explanation-generation tokens (reserved for FR-9). If the projected
cost exceeds the $0.08 guardrail, the caller must degrade to the human path
rather than spend beyond budget. All prices are illustrative POC figures.
"""

from __future__ import annotations

COST_CEILING_USD = 0.08

# Illustrative unit prices (USD).
_FIXED_OVERHEAD = 0.002
_PER_RETRIEVED_CLAUSE = 0.0005
_MODEL_INFERENCE = 0.001
_PER_REGULATORY_CHECK = 0.004
_DOCUMENT_CHECK = 0.001
_LLM_PER_1K_TOKENS = 0.03  # applies once an LLM is wired for explanations (FR-9)


def estimate_explanation_tokens(num_clauses: int, num_factors: int) -> int:
    """Rough token projection for a grounded explanation over the evidence."""
    return 300 + 60 * num_clauses + 30 * num_factors


def estimate_cost(
    *,
    num_retrieved_clauses: int = 0,
    num_regulatory_checks: int = 0,
    ran_model_inference: bool = True,
    ran_document_check: bool = True,
    projected_explanation_tokens: int = 0,
) -> dict:
    """Return {cost_usd, breached, breakdown} for one assessment."""
    breakdown = {
        "fixed_overhead": _FIXED_OVERHEAD,
        "retrieval": round(num_retrieved_clauses * _PER_RETRIEVED_CLAUSE, 6),
        "model_inference": _MODEL_INFERENCE if ran_model_inference else 0.0,
        "regulatory": round(num_regulatory_checks * _PER_REGULATORY_CHECK, 6),
        "documents": _DOCUMENT_CHECK if ran_document_check else 0.0,
        "explanation": round((projected_explanation_tokens / 1000) * _LLM_PER_1K_TOKENS, 6),
    }
    cost = round(sum(breakdown.values()), 6)
    return {"cost_usd": cost, "breached": cost > COST_CEILING_USD, "breakdown": breakdown}

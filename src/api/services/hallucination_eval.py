"""Hallucination & faithfulness eval harness (AC-7 / US-404).

For each generated explanation, every claim must be grounded in a source that
actually exists in that decision's evidence chain. A claim whose source is
absent from the evidence is a hallucination. The harness computes, over a batch
of decisions:

  - faithfulness  = grounded_claims / total_claims       (target >= 0.90)
  - hallucination_rate = 1 - faithfulness                 (target < 0.01)

and blocks the release when either bound is breached. The "judge" here is a
deterministic source-grounding check; it is the same contract an LLM judge would
enforce, so an LLM judge can be dropped in without changing the harness.
"""

from __future__ import annotations

from .explanation import generate_explanation

FAITHFULNESS_FLOOR = 0.90
HALLUCINATION_CEILING = 0.01


def _valid_sources(evidence_chain: dict) -> set[str]:
    """The set of source ids/keys a claim may legitimately cite for this decision."""
    sources = {
        "risk_score", "regulatory_result", "document_findings", "recommendation",
    }
    for factor in evidence_chain.get("risk_factors") or []:
        sources.add(f"risk_factor:{factor['factor']}")
    for clause in evidence_chain.get("policy_clauses") or []:
        sources.add(clause.get("source_id") or clause["clause_id"])
        sources.add(clause["clause_id"])
    for rule in evidence_chain.get("policy_failed_rules") or []:
        sources.add(rule["clause_id"])
    return sources


def evaluate_explanation(evidence_chain: dict, recommendation: str) -> dict:
    explanation = generate_explanation(evidence_chain, recommendation)
    valid = _valid_sources(evidence_chain)
    total = len(explanation["claims"])
    grounded = sum(1 for c in explanation["claims"] if c["source"] in valid)
    hallucinated = [c for c in explanation["claims"] if c["source"] not in valid]
    return {
        "total_claims": total,
        "grounded_claims": grounded,
        "hallucinated_claims": hallucinated,
        "faithfulness": round(grounded / total, 4) if total else 1.0,
    }


def run_eval(cases: list[dict]) -> dict:
    """Evaluate a batch of {evidence_chain, recommendation} cases.

    Returns aggregate faithfulness + hallucination rate and whether the release
    is blocked (US-404 AC: build blocked on regression).
    """
    total_claims = grounded_claims = 0
    per_case = []
    for case in cases:
        result = evaluate_explanation(case["evidence_chain"], case["recommendation"])
        total_claims += result["total_claims"]
        grounded_claims += result["grounded_claims"]
        per_case.append(result)

    faithfulness = round(grounded_claims / total_claims, 4) if total_claims else 1.0
    hallucination_rate = round(1 - faithfulness, 4)
    blocked = faithfulness < FAITHFULNESS_FLOOR or hallucination_rate >= HALLUCINATION_CEILING

    return {
        "n_cases": len(cases),
        "total_claims": total_claims,
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "faithfulness_floor": FAITHFULNESS_FLOOR,
        "hallucination_ceiling": HALLUCINATION_CEILING,
        "release_blocked": blocked,
        "cases": per_case,
    }

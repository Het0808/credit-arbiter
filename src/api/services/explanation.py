"""Grounded explanation generation (FR-9 stand-in for US-208 / feeds US-404).

Generates a narrative recommendation explanation composed ONLY from the stored
evidence chain: every sentence is emitted together with the source it is
grounded in (a policy clause source_id, an applicant field, a risk factor, the
regulatory verdict, or a document finding). Because the text is assembled from
evidence rather than free-generated, it is grounded by construction - which is
exactly what the faithfulness harness (hallucination_eval.py) verifies.

When an LLM is later adopted for richer prose, it must consume the same
``claims`` structure and its output must pass the same harness before release.
"""

from __future__ import annotations


def generate_explanation(evidence_chain: dict, recommendation: str) -> dict:
    """Return {narrative, claims} where each claim carries its grounding source."""
    claims: list[dict] = []

    risk_band = evidence_chain.get("risk_band")
    risk_score = evidence_chain.get("risk_score")
    if risk_band is not None and risk_score is not None:
        claims.append({
            "text": f"The model assessed this application as {risk_band} risk "
                    f"({risk_score * 100:.1f}% probability of default).",
            "source": "risk_score",
        })

    for factor in (evidence_chain.get("risk_factors") or [])[:3]:
        claims.append({
            "text": f"{factor['label']} {factor['direction'].replace('_', ' ')} "
                    f"(contribution {factor['contribution']}).",
            "source": f"risk_factor:{factor['factor']}",
        })

    for clause in evidence_chain.get("policy_clauses") or []:
        claims.append({
            "text": f"Policy clause {clause['clause_id']} "
                    f"({clause.get('corpus_version')}) was applied.",
            "source": clause.get("source_id") or clause["clause_id"],
        })

    failed = evidence_chain.get("policy_failed_rules") or []
    for rule in failed:
        claims.append({
            "text": f"Policy rule {rule['clause_id']} was not satisfied "
                    f"(action: {rule['on_violation']}).",
            "source": rule["clause_id"],
        })

    reg_status = evidence_chain.get("regulatory_status")
    if reg_status is not None:
        claims.append({
            "text": f"Regulatory verification returned {reg_status}.",
            "source": "regulatory_result",
        })

    docs = evidence_chain.get("document_findings") or {}
    if docs:
        if docs.get("missing_information"):
            claims.append({
                "text": f"Required documents are missing: {', '.join(docs['missing_information'])}.",
                "source": "document_findings",
            })
        else:
            claims.append({"text": "All required documents are present.", "source": "document_findings"})

    claims.append({
        "text": f"Recommendation: {recommendation}.",
        "source": "recommendation",
    })

    narrative = " ".join(c["text"] for c in claims)
    return {"narrative": narrative, "claims": claims, "recommendation": recommendation}

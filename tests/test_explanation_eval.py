"""Tests for grounded explanation + hallucination eval harness (US-404)."""

from src.api.services.explanation import generate_explanation
from src.api.services.hallucination_eval import evaluate_explanation, run_eval

EVIDENCE = {
    "risk_score": 0.12,
    "risk_band": "Low",
    "risk_factors": [
        {"factor": "dti", "label": "Debt-to-income ratio", "contribution": 0.05, "direction": "increases_risk"},
    ],
    "policy_clauses": [
        {"clause_id": "POL-PL-001", "source_id": "POL-PL-001", "corpus_version": "v1.0"},
    ],
    "policy_failed_rules": [],
    "regulatory_status": "PASS",
    "document_findings": {"missing_information": [], "verified": True},
}


def test_generated_explanation_is_fully_grounded():
    result = evaluate_explanation(EVIDENCE, "Approve")
    assert result["faithfulness"] == 1.0
    assert result["hallucinated_claims"] == []


def test_narrative_cites_sources():
    explanation = generate_explanation(EVIDENCE, "Approve")
    assert "POL-PL-001" in explanation["narrative"]
    assert all("source" in c for c in explanation["claims"])


def test_run_eval_passes_for_grounded_batch():
    cases = [{"evidence_chain": EVIDENCE, "recommendation": "Approve"} for _ in range(5)]
    report = run_eval(cases)
    assert report["faithfulness"] >= 0.90
    assert report["hallucination_rate"] < 0.01
    assert report["release_blocked"] is False


def test_run_eval_blocks_release_on_hallucination():
    # A claim citing a source not in the evidence must count as a hallucination.
    from src.api.services import hallucination_eval as he

    def _fake_generate(evidence_chain, recommendation):
        return {
            "narrative": "x",
            "claims": [
                {"text": "grounded", "source": "risk_score"},
                {"text": "made up", "source": "POL-NONEXISTENT-999"},
            ],
            "recommendation": recommendation,
        }

    original = he.generate_explanation
    he.generate_explanation = _fake_generate
    try:
        report = he.run_eval([{"evidence_chain": EVIDENCE, "recommendation": "Approve"}])
    finally:
        he.generate_explanation = original
    assert report["release_blocked"] is True
    assert report["hallucination_rate"] > 0

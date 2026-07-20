"""Tests for the production integrations: vector RAG, LLM explanation, config switches."""

import pytest

from src.api.services.explanation import generate_explanation
from src.api.services.llm_explanation import llm_enabled

EVIDENCE = {
    "risk_score": 0.12, "risk_band": "Low",
    "risk_factors": [{"factor": "dti", "label": "Debt-to-income ratio", "contribution": 0.05,
                      "direction": "increases_risk"}],
    "policy_clauses": [{"clause_id": "POL-PL-001", "source_id": "POL-PL-001", "corpus_version": "v1.0"}],
    "policy_failed_rules": [], "regulatory_status": "PASS",
    "document_findings": {"missing_information": [], "verified": True},
}


# --- LLM explanation (fallback when disabled) ---

def test_llm_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_enabled() is False


def test_explanation_falls_back_to_deterministic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    result = generate_explanation(EVIDENCE, "Approve")
    assert result["generator"] == "deterministic"
    assert "POL-PL-001" in result["narrative"]
    assert result["claims"]


def test_llm_enabled_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm_enabled() is False  # provider set but no key -> disabled


# --- Vector RAG (semantic retrieval) ---

def test_vector_retrieval_is_semantic(monkeypatch):
    pytest.importorskip("qdrant_client")
    pytest.importorskip("fastembed")
    monkeypatch.setenv("RETRIEVER", "vector")
    from src.api.services.retrieval import retrieve

    # A paraphrase with little keyword overlap should still hit affordability clauses.
    result = retrieve("can the borrower keep up with monthly repayments on their wages?",
                      scheme="Personal Loan")
    assert result["retrieval_failed"] is False
    assert all(c["clause_id"].startswith("POL-PL-") for c in result["clauses"])
    assert all(c["corpus_version"] == "v1.0" for c in result["clauses"])


def test_vector_scheme_filter(monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("RETRIEVER", "vector")
    from src.api.services.retrieval import retrieve

    result = retrieve("loan to income cap", scheme="Vehicle Loan")
    assert all(c["scheme"] == "Vehicle Loan" for c in result["clauses"])

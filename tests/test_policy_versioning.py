"""Tests for scheme-aware retrieval + policy version management (US-204, US-207)."""

from src.api.services import retrieval as R


def setup_function():
    R.reindex("v1.0")  # ensure a known active version for each test


def test_scheme_filter_restricts_candidates():
    out = R.retrieve("loan to income cap", scheme="Vehicle Loan", top_k=5)
    assert out["retrieval_failed"] is False
    assert all(c["scheme"] == "Vehicle Loan" for c in out["clauses"])
    assert all(c["clause_id"].startswith("POL-VH-") for c in out["clauses"])


def test_every_clause_has_source_id_and_version():
    out = R.retrieve("minimum income", scheme="Personal Loan")
    for clause in out["clauses"]:
        assert clause["source_id"] == clause["clause_id"]
        assert clause["corpus_version"] == "v1.0"


def test_unknown_scheme_fails_retrieval():
    out = R.retrieve("anything", scheme="Mortgage")
    assert out["retrieval_failed"] is True
    assert out["clauses"] == []


def test_versions_registry_lists_both_corpora():
    versions = R.list_versions()
    assert "v0.1" in versions["available_versions"]
    assert "v1.0" in versions["available_versions"]
    assert versions["active_version"] == "v1.0"


def test_replay_against_old_version():
    out = R.retrieve("maximum debt to income personal loan", corpus_version="v0.1")
    assert out["clauses"][0]["corpus_version"] == "v0.1"
    assert out["clauses"][0]["clause_id"] == "POL-PL-001"


def test_reindex_switches_active_version():
    meta = R.reindex("v0.1")
    assert meta["corpus_version"] == "v0.1"
    assert R.active_version() == "v0.1"
    R.reindex("v1.0")  # restore
    assert R.active_version() == "v1.0"

import json

from src.api.services.retrieval import POLICY_CORPUS_PATH, reindex_corpus, retrieve


def test_relevant_query_returns_expected_clause_with_source_id():
    result = retrieve("What is the maximum debt-to-income ratio for a personal loan?", scheme="Personal")
    assert result["retrieval_failed"] is False
    assert len(result["clauses"]) >= 1
    assert result["clauses"][0]["clause_id"] == "POL-PL-001"
    assert result["clauses"][0]["source_id"] == "POL-PL-001"
    assert result["clauses"][0]["scheme"] == "Personal"
    assert 0.0 < result["clauses"][0]["score"] <= 1.0


def test_unrelated_query_flags_retrieval_failed():
    result = retrieve("What is the weather forecast for tomorrow in outer space?")
    assert result["retrieval_failed"] is True
    assert result["clauses"] == []


def test_thin_file_query_matches_escalation_clause():
    result = retrieve("applicant has thin file short employment history bureau record")
    assert result["retrieval_failed"] is False
    assert result["clauses"][0]["clause_id"] == "POL-PL-004"


def test_scheme_filter_restricts_candidates_to_that_scheme():
    result = retrieve("what is the debt to income threshold", scheme="Education")
    assert result["retrieval_failed"] is False
    assert all(clause["scheme"] == "Education" for clause in result["clauses"])
    assert result["clauses"][0]["clause_id"] == "POL-ED-001"


def test_unknown_scheme_returns_retrieval_failed():
    result = retrieve("debt to income", scheme="Nonexistent Scheme")
    assert result["retrieval_failed"] is True
    assert result["clauses"] == []


def test_reindex_corpus_activates_the_new_version(tmp_path):
    """US-207: a manual re-index makes a new corpus version live, without a
    restart, in the same running process."""
    new_corpus = {
        "version": "v1.1-test",
        "effective_date": "2026-07-12",
        "schemes": ["Personal"],
        "clauses": [
            {
                "clause_id": "POL-PL-999",
                "scheme": "Personal",
                "title": "New Test Clause",
                "text": "A brand new clause about the maximum debt to income ratio.",
            }
        ],
    }
    new_path = tmp_path / "corpus_v1_1.json"
    new_path.write_text(json.dumps(new_corpus), encoding="utf-8")

    try:
        metadata = reindex_corpus(str(new_path))
        assert metadata["version"] == "v1.1-test"
        assert metadata["clause_count"] == 1

        result = retrieve("maximum debt to income ratio", scheme="Personal")
        assert result["clauses"][0]["clause_id"] == "POL-PL-999"
        assert result["clauses"][0]["version"] == "v1.1-test"
    finally:
        reindex_corpus(POLICY_CORPUS_PATH)

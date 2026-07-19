from src.api.services.retrieval import retrieve


def test_relevant_query_returns_expected_clause_with_source_id():
    result = retrieve("What is the maximum debt-to-income ratio for a personal loan?", scheme="Personal Loan")
    assert result["retrieval_failed"] is False
    assert len(result["clauses"]) >= 1
    top = result["clauses"][0]
    assert top["clause_id"] == "POL-PL-001"
    assert top["source_id"] == "POL-PL-001"
    assert top["corpus_version"] == result["corpus_version"]
    assert 0.0 < top["score"] <= 1.0


def test_unrelated_query_flags_retrieval_failed():
    result = retrieve("What is the weather forecast for tomorrow in outer space?")
    assert result["retrieval_failed"] is True
    assert result["clauses"] == []


def test_thin_file_query_matches_escalation_clause():
    result = retrieve(
        "applicant has thin file short employment history bureau record", scheme="Personal Loan"
    )
    assert result["retrieval_failed"] is False
    assert result["clauses"][0]["clause_id"] == "POL-PL-004"

"""Tests for retrieval quality monitoring (US-205)."""

from src.api.services.retrieval_monitor import evaluate_retrieval_quality


def test_eval_set_meets_precision_recall_floors():
    report = evaluate_retrieval_quality()
    assert report["context_precision"] >= 0.85
    assert report["context_recall"] >= 0.85
    assert report["retrieval_failure_rate"] <= 0.05
    assert report["healthy"] is True
    assert report["alerts"] == []


def test_degraded_retrieval_raises_alert():
    # An eval set of impossible queries should breach the failure-rate ceiling.
    bogus = [
        {"scheme": "Personal Loan", "query": "zzzz qqqq xxxx nonsense token", "expected_clause_id": "POL-PL-001"},
        {"scheme": "Personal Loan", "query": "blorp glorp znak wibble", "expected_clause_id": "POL-PL-002"},
    ]
    report = evaluate_retrieval_quality(eval_set=bogus)
    assert report["healthy"] is False
    assert report["alerts"]

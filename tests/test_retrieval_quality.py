"""US-205: retrieval quality monitoring. A pragmatic stand-in for RAGAS
context precision/recall (single-relevant-doc-per-query eval against the
curated eval set, not the full LLM-judge RAGAS metric) - same documented-
simplification style as US-404's grounding check."""

from src.api.services.retrieval_quality import evaluate_retrieval_quality


def _fake_retrieve(clauses_by_query, failed_queries=()):
    def retrieve_fn(query, scheme=None):
        if query in failed_queries:
            return {"clauses": [], "retrieval_failed": True}
        return {"clauses": clauses_by_query[query], "retrieval_failed": False}

    return retrieve_fn


def test_perfect_eval_set_scores_1_0():
    eval_set = [
        {"query": "q1", "expected_clause_id": "POL-A-001"},
        {"query": "q2", "expected_clause_id": "POL-A-002"},
    ]
    retrieve_fn = _fake_retrieve(
        {
            "q1": [{"clause_id": "POL-A-001"}, {"clause_id": "POL-A-999"}],
            "q2": [{"clause_id": "POL-A-002"}],
        }
    )

    result = evaluate_retrieval_quality(eval_set, retrieve_fn=retrieve_fn)

    assert result == {"precision": 1.0, "recall": 1.0, "failure_rate": 0.0, "n": 2}


def test_wrong_top1_but_present_lower_in_topk_hurts_precision_not_recall():
    eval_set = [{"query": "q1", "expected_clause_id": "POL-A-001"}]
    retrieve_fn = _fake_retrieve(
        {"q1": [{"clause_id": "POL-A-999"}, {"clause_id": "POL-A-001"}]}
    )

    result = evaluate_retrieval_quality(eval_set, retrieve_fn=retrieve_fn)

    assert result["precision"] == 0.0
    assert result["recall"] == 1.0


def test_retrieval_failure_counts_against_precision_recall_and_failure_rate():
    eval_set = [
        {"query": "q1", "expected_clause_id": "POL-A-001"},
        {"query": "q2", "expected_clause_id": "POL-A-002"},
    ]
    retrieve_fn = _fake_retrieve(
        {"q1": [{"clause_id": "POL-A-001"}], "q2": []}, failed_queries={"q2"}
    )

    result = evaluate_retrieval_quality(eval_set, retrieve_fn=retrieve_fn)

    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["failure_rate"] == 0.5


def test_real_eval_set_meets_the_85_percent_floor():
    """The actual US-205 acceptance check, run against the real corpus."""
    result = evaluate_retrieval_quality()
    assert result["precision"] >= 0.85, result
    assert result["recall"] >= 0.85, result
    assert result["failure_rate"] <= 0.05, result

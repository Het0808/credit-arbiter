"""US-111 spike: policy RAG retrieval accuracy, scaled to the single-scheme
5-clause POC corpus (the PRD's full de-risk spike uses 30 scenarios across
multiple schemes; this is a regression-check-sized version of the same idea,
not a production-scale retrieval-quality claim - see docs/demo_script.md)."""

import json
import os

from src.api.services.retrieval import retrieve

EVAL_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval", "retrieval_eval_set.json"
)

PASS_THRESHOLD = 0.85


def _load_eval_set():
    with open(EVAL_SET_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_retrieval_accuracy_meets_85_percent_floor():
    eval_set = _load_eval_set()
    results = []
    for case in eval_set:
        outcome = retrieve(case["query"], scheme=case.get("scheme"))
        top_clause_id = outcome["clauses"][0]["clause_id"] if outcome["clauses"] else None
        passed = top_clause_id == case["expected_clause_id"]
        results.append((case["query"], case["expected_clause_id"], top_clause_id, passed))

    pass_rate = sum(1 for r in results if r[3]) / len(results)

    if pass_rate < PASS_THRESHOLD:
        print("\nRetrieval accuracy failures:")
        for query, expected, actual, passed in results:
            if not passed:
                print(f"  FAIL query={query!r} expected={expected} actual={actual}")

    assert pass_rate >= PASS_THRESHOLD, f"retrieval accuracy {pass_rate:.0%} below {PASS_THRESHOLD:.0%} floor"

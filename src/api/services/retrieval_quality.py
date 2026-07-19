"""Retrieval quality monitoring (US-205).

Pragmatic stand-in for RAGAS context precision/recall - a single-relevant-
clause-per-query eval against the curated eval set, not the full LLM-judge
RAGAS metric (same documented-simplification style as US-404's grounding
check). precision = top-1 hit rate (matches how assessment.py actually
consumes retrieval - only clauses[0] drives the recommendation). recall =
hit rate anywhere in the returned candidate set. failure_rate = fraction of
queries retrieval flagged as failed.

Run as the "daily eval job" via `python -m scripts.retrieval_quality_report`.
"""

import json
import os

from .retrieval import retrieve

EVAL_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data",
    "eval",
    "retrieval_eval_set.json",
)

PRECISION_RECALL_FLOOR = 0.85
FAILURE_RATE_ALERT_THRESHOLD = 0.05


def _load_eval_set():
    with open(EVAL_SET_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_retrieval_quality(eval_set=None, retrieve_fn=None):
    if eval_set is None:
        eval_set = _load_eval_set()
    if retrieve_fn is None:
        retrieve_fn = lambda query, scheme=None: retrieve(query, scheme=scheme)

    precision_hits = 0
    recall_hits = 0
    failures = 0
    for case in eval_set:
        outcome = retrieve_fn(case["query"], scheme=case.get("scheme"))
        clause_ids = [c["clause_id"] for c in outcome["clauses"]]
        if outcome["retrieval_failed"]:
            failures += 1
        if clause_ids and clause_ids[0] == case["expected_clause_id"]:
            precision_hits += 1
        if case["expected_clause_id"] in clause_ids:
            recall_hits += 1

    n = len(eval_set)
    return {
        "precision": round(precision_hits / n, 4),
        "recall": round(recall_hits / n, 4),
        "failure_rate": round(failures / n, 4),
        "n": n,
    }

"""Retrieval quality monitoring (US-205).

Computes context precision, context recall, and the retrieval failure rate over
a labelled eval set, and raises an alert when quality degrades:
- context precision (top-1 correct)  must stay >= 0.85
- context recall    (expected clause within top-k)  must stay >= 0.85
- retrieval failure rate  must stay <= 5%  (AC: alert when > 5%)

Designed to run as a scheduled daily job (see scripts/retrieval_quality_report.py)
and as an in-process check in tests. Uses the labelled eval set as the ground
truth for relevance; a RAGAS-style embedding judge can replace the exact-match
relevance check once an embedding model is adopted (Sprint 2+ upgrade path).
"""

from __future__ import annotations

import json
import os

from .retrieval import retrieve

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_EVAL_PATH = os.path.join(REPO_ROOT, "data", "eval", "retrieval_eval_set.json")

PRECISION_FLOOR = 0.85
RECALL_FLOOR = 0.85
FAILURE_RATE_CEILING = 0.05


def load_eval_set(path: str = DEFAULT_EVAL_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_retrieval_quality(eval_set: list[dict] | None = None, top_k: int = 3) -> dict:
    """Run every eval case through scheme-aware retrieval and score quality.

    Returns a report dict with metrics, per-case rows, and a list of triggered
    alerts (empty when everything is within thresholds).
    """
    if eval_set is None:
        eval_set = load_eval_set()

    rows, hits_at_1, recall_hits, failures = [], 0, 0, 0
    for case in eval_set:
        expected = case["expected_clause_id"]
        outcome = retrieve(case["query"], scheme=case.get("scheme"), top_k=top_k)
        retrieved_ids = [c["clause_id"] for c in outcome["clauses"]]

        failed = outcome["retrieval_failed"] or not retrieved_ids
        top1 = retrieved_ids[0] if retrieved_ids else None
        in_topk = expected in retrieved_ids

        failures += int(failed)
        hits_at_1 += int(top1 == expected)
        recall_hits += int(in_topk)

        rows.append(
            {
                "scheme": case.get("scheme"),
                "query": case["query"],
                "expected": expected,
                "top1": top1,
                "in_topk": in_topk,
                "retrieval_failed": failed,
            }
        )

    n = len(eval_set) or 1
    precision = round(hits_at_1 / n, 4)
    recall = round(recall_hits / n, 4)
    failure_rate = round(failures / n, 4)

    alerts = []
    if precision < PRECISION_FLOOR:
        alerts.append(f"context_precision {precision:.0%} below {PRECISION_FLOOR:.0%} floor")
    if recall < RECALL_FLOOR:
        alerts.append(f"context_recall {recall:.0%} below {RECALL_FLOOR:.0%} floor")
    if failure_rate > FAILURE_RATE_CEILING:
        alerts.append(f"retrieval_failure_rate {failure_rate:.0%} above {FAILURE_RATE_CEILING:.0%} ceiling")

    return {
        "n_cases": len(eval_set),
        "top_k": top_k,
        "context_precision": precision,
        "context_recall": recall,
        "retrieval_failure_rate": failure_rate,
        "alerts": alerts,
        "healthy": not alerts,
        "cases": rows,
    }

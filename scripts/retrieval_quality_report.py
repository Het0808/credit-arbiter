"""US-205 daily eval job: reports retrieval context precision/recall against
the curated eval set and exits non-zero if either falls below the 0.85 floor
or the failure rate exceeds the 5% alert threshold. Stdlib + existing deps
only - no new eval framework.

    python -m scripts.retrieval_quality_report
"""

import sys

from src.api.services.retrieval_quality import (
    FAILURE_RATE_ALERT_THRESHOLD,
    PRECISION_RECALL_FLOOR,
    evaluate_retrieval_quality,
)


def main():
    result = evaluate_retrieval_quality()
    print(
        f"precision={result['precision']:.2%} recall={result['recall']:.2%} "
        f"failure_rate={result['failure_rate']:.2%} (n={result['n']})"
    )

    ok = True
    if result["precision"] < PRECISION_RECALL_FLOOR:
        print(f"ALERT: precision below {PRECISION_RECALL_FLOOR:.0%} floor")
        ok = False
    if result["recall"] < PRECISION_RECALL_FLOOR:
        print(f"ALERT: recall below {PRECISION_RECALL_FLOOR:.0%} floor")
        ok = False
    if result["failure_rate"] > FAILURE_RATE_ALERT_THRESHOLD:
        print(f"ALERT: retrieval failure rate above {FAILURE_RATE_ALERT_THRESHOLD:.0%} threshold")
        ok = False

    if not ok:
        sys.exit(1)
    print("Retrieval quality within bounds.")


if __name__ == "__main__":
    main()

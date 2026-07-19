"""Daily retrieval-quality report (US-205).

Run from the repo root (venv activated):

    python -m scripts.retrieval_quality_report

Evaluates scheme-aware retrieval over data/eval/retrieval_eval_set.json and
writes reports/ml/retrieval_quality.json. Exits non-zero when any quality
threshold is breached (context precision < 0.85, recall < 0.85, or failure
rate > 5%), so it can gate a scheduled/CI job and fire an alert.
"""

import json
import os
import sys

from src.api.services.retrieval_monitor import evaluate_retrieval_quality

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(REPO_ROOT, "reports", "ml", "retrieval_quality.json")


def main() -> int:
    report = evaluate_retrieval_quality()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"context_precision : {report['context_precision']:.2%}")
    print(f"context_recall    : {report['context_recall']:.2%}")
    print(f"failure_rate      : {report['retrieval_failure_rate']:.2%}")
    print(f"report written    : {REPORT_PATH}")

    if report["alerts"]:
        print("\nALERT - retrieval quality degraded:")
        for alert in report["alerts"]:
            print(f"  - {alert}")
        return 1

    print("\nretrieval quality healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())

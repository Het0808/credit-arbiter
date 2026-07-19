"""Underwriter pilot simulation (AC-11 / US-408).

Simulates the pilot protocol - 3 underwriters x 20 files each - by generating 60
varied synthetic applications, running each through the copilot, and simulating
underwriter accept/override behaviour and review time. Reports acceptance rate
(target >= 75%) and median review time (target <= 22 min), and produces an
override remediation list.

This is a SIMULATION (no real underwriters); the numbers are illustrative of the
protocol and pipeline, not a real pilot result. Real US-408 sign-off requires
three human underwriters running the protocol against this same build.

Run:  python -m scripts.run_pilot
"""

import json
import random
import statistics
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database import Base
from src.api.models import Application, DecisionRecord
from src.api.services.assessment import run_assessment

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "reports" / "ops" / "pilot_results.json"
REPORT_MD = REPO_ROOT / "docs" / "PILOT_RESULTS.md"

UNDERWRITERS = ["uw_alice", "uw_bob", "uw_carol"]
FILES_PER_UW = 20
SEED = 42

OVERRIDE_REASONS = ["compensating_factors", "additional_documentation", "data_quality_issue", "risk_appetite"]


def _session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _make_application(db, i, rng):
    income = rng.choice([90000, 130000, 180000, 250000, 320000])
    credit = income * rng.choice([1.0, 2.5, 4.0, 6.5])
    annuity = income * rng.choice([0.15, 0.3, 0.45, 0.6])
    tenure = rng.choice([-200, -800, -1800, -3650, -6000])
    app = Application(
        external_id=f"pilot-{i}", loan_scheme="Personal Loan", amt_income_total=income,
        amt_credit=credit, amt_annuity=annuity, days_employed=tenure,
        days_birth=-rng.randint(8000, 22000), code_gender=rng.choice(["M", "F"]), status="COMPLETE",
    )
    db.add(app); db.commit(); db.refresh(app)
    return app


def _simulate_underwriter(record, rng):
    """Return (action, reason_code, review_minutes) for a simulated underwriter."""
    if record.escalation_flag:
        # Escalated cases take longer and are accepted less often.
        review = rng.uniform(15, 24)
        if rng.random() < 0.55:
            return "accept", None, review
        return "override", rng.choice(OVERRIDE_REASONS), review
    # Clear recommendations: quick, usually accepted.
    review = rng.uniform(6, 16)
    if rng.random() < 0.92:
        return "accept", None, review
    return "override", rng.choice(OVERRIDE_REASONS), review


def main():
    rng = random.Random(SEED)
    db = _session()

    rows, review_times, overrides = [], [], []
    accepts = 0
    idx = 0
    for uw in UNDERWRITERS:
        for _ in range(FILES_PER_UW):
            app = _make_application(db, idx, rng); idx += 1
            record = run_assessment(db, app)
            action, reason_code, minutes = _simulate_underwriter(record, rng)
            record.underwriter_action = action
            record.underwriter_reason_code = reason_code
            record.underwriter_action_at = datetime.utcnow()
            db.commit()

            review_times.append(minutes)
            if action == "accept":
                accepts += 1
            else:
                overrides.append(reason_code)
            rows.append({"underwriter": uw, "recommendation": record.recommendation,
                         "action": action, "reason_code": reason_code, "review_min": round(minutes, 1)})

    total = len(rows)
    acceptance_rate = round(accepts / total, 4)
    median_review = round(statistics.median(review_times), 1)

    remediation = {}
    for r in overrides:
        remediation[r] = remediation.get(r, 0) + 1
    remediation_list = sorted(remediation.items(), key=lambda kv: kv[1], reverse=True)

    report = {
        "simulation": True,
        "underwriters": len(UNDERWRITERS),
        "files_per_underwriter": FILES_PER_UW,
        "total_files": total,
        "acceptance_rate": acceptance_rate,
        "acceptance_target": 0.75,
        "acceptance_met": acceptance_rate >= 0.75,
        "median_review_minutes": median_review,
        "review_time_target_min": 22,
        "review_time_met": median_review <= 22,
        "override_remediation": [{"reason_code": k, "count": v} for k, v in remediation_list],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2))

    md = [
        "# Underwriter Pilot Results (US-408) — SIMULATED",
        "",
        "> This is a simulation of the 3x20 pilot protocol, not a real pilot. Real sign-off",
        "> requires three human underwriters running the protocol against this build.",
        "",
        f"- Files processed: **{total}** ({len(UNDERWRITERS)} underwriters x {FILES_PER_UW})",
        f"- Acceptance rate: **{acceptance_rate:.0%}** (target ≥ 75% → {'PASS' if report['acceptance_met'] else 'FAIL'})",
        f"- Median review time: **{median_review} min** (target ≤ 22 min → {'PASS' if report['review_time_met'] else 'FAIL'})",
        "",
        "## Override remediation list",
        "",
        "| Reason code | Count |",
        "|---|---|",
    ]
    md += [f"| {k} | {v} |" for k, v in remediation_list] or ["| (none) | 0 |"]
    REPORT_MD.write_text("\n".join(md) + "\n")

    print(json.dumps(report, indent=2))
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Operations dashboard KPIs (PRD §8 / US-407).

Aggregates the live decision history (plus the latest load-test report, if
present) into the PRD §8 KPIs with do-not-ship thresholds and alerts:
throughput, P95 latency, cost/app, acceptance rate, override rate, fairness gap.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from ..models import DecisionRecord
from .fairness_monitor import run_fairness_monitor

REPO_ROOT = Path(__file__).resolve().parents[3]
LOAD_TEST_REPORT = REPO_ROOT / "reports" / "ops" / "load_test.json"

# Do-not-ship thresholds (PRD §8/§12).
THRESHOLDS = {
    "p95_latency_s": 20.0,
    "cost_per_app_usd": 0.08,
    "fairness_gap_pp": 5.0,
    "acceptance_rate_min": 0.75,
}


def _p95_latency() -> float | None:
    if LOAD_TEST_REPORT.exists():
        try:
            return json.loads(LOAD_TEST_REPORT.read_text()).get("p95_latency_s")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def compute_dashboard(db: Session, window_days: int = 7) -> dict:
    since = datetime.utcnow() - timedelta(days=window_days)
    decisions = db.query(DecisionRecord).filter(DecisionRecord.created_at >= since).all()
    n = len(decisions)

    actioned = [d for d in decisions if d.underwriter_action]
    overrides = [d for d in actioned if d.underwriter_action == "override"]
    accepts = [d for d in actioned if d.underwriter_action == "accept"]
    costs = [d.estimated_cost_usd for d in decisions if d.estimated_cost_usd is not None]

    fairness = run_fairness_monitor(db, enforce=False)  # read-only: dashboard must not pause schemes
    max_gap = max((r["max_delta_pp"] for r in fairness["schemes"].values()), default=0.0)

    p95 = _p95_latency()
    cost_per_app = round(sum(costs) / len(costs), 6) if costs else 0.0
    acceptance_rate = round(len(accepts) / len(actioned), 4) if actioned else None
    override_rate = round(len(overrides) / len(actioned), 4) if actioned else None

    kpis = {
        "throughput_decisions": n,
        "window_days": window_days,
        "p95_latency_s": p95,
        "cost_per_app_usd": cost_per_app,
        "acceptance_rate": acceptance_rate,
        "override_rate": override_rate,
        "fairness_gap_pp": max_gap,
    }

    alerts = []
    if p95 is not None and p95 > THRESHOLDS["p95_latency_s"]:
        alerts.append(f"P95 latency {p95}s exceeds {THRESHOLDS['p95_latency_s']}s")
    if cost_per_app > THRESHOLDS["cost_per_app_usd"]:
        alerts.append(f"cost/app ${cost_per_app} exceeds ${THRESHOLDS['cost_per_app_usd']}")
    if max_gap > THRESHOLDS["fairness_gap_pp"]:
        alerts.append(f"fairness gap {max_gap}pp exceeds {THRESHOLDS['fairness_gap_pp']}pp")
    if acceptance_rate is not None and acceptance_rate < THRESHOLDS["acceptance_rate_min"]:
        alerts.append(f"acceptance rate {acceptance_rate:.0%} below {THRESHOLDS['acceptance_rate_min']:.0%}")

    return {"kpis": kpis, "thresholds": THRESHOLDS, "alerts": alerts, "healthy": not alerts}

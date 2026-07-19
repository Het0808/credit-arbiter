import logging
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import DecisionRecord, User
from ..services.fairness_check import current_max_fairness_gap_pp
from ..services.retrieval_quality import FAILURE_RATE_ALERT_THRESHOLD

router = APIRouter(prefix="/metrics", tags=["metrics"])
logger = logging.getLogger(__name__)

# PRD thresholds this dashboard alerts against (AC-5, AC-8, AC-9, US-205).
AVG_COST_ALERT_USD = 0.05
P95_LATENCY_ALERT_MS = 20_000
FAIRNESS_GAP_ALERT_PP = 5.0


@router.get("")
def get_metrics(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Ops dashboard aggregate (US-407, + US-205 live retrieval failure
    rate): throughput, acceptance rate, override rate, avg cost, avg
    latency, current fairness gap status. Any threshold breach is logged
    as a WARNING (log-based alert - no email/Slack integration in this POC,
    same minimal style as audit_log.py)."""
    records = db.query(DecisionRecord).all()
    total = len(records)

    decided = [r for r in records if r.underwriter_action is not None]
    accepted = [r for r in decided if r.underwriter_action == "accept"]
    overridden = [r for r in decided if r.underwriter_action == "override"]

    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    latencies = [r.latency_ms for r in records if r.latency_ms is not None]

    recommendation_counts = dict(Counter(r.recommendation for r in records))

    escalation_count = sum(1 for r in records if r.escalation_flag)

    retrieval_failure_rate = round(sum(1 for r in records if r.retrieval_failed) / total, 4) if total else None
    avg_cost_usd = round(sum(costs) / len(costs), 6) if costs else None
    p95_latency_ms = round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else None
    fairness_gap_pp = current_max_fairness_gap_pp()

    alerts = {
        "retrieval_failure_alert": (retrieval_failure_rate or 0.0) > FAILURE_RATE_ALERT_THRESHOLD,
        "avg_cost_alert": (avg_cost_usd or 0.0) > AVG_COST_ALERT_USD,
        "p95_latency_alert": (p95_latency_ms or 0.0) > P95_LATENCY_ALERT_MS,
        "fairness_gap_alert": (fairness_gap_pp or 0.0) > FAIRNESS_GAP_ALERT_PP,
    }
    for name, breached in alerts.items():
        if breached:
            logger.warning("ops dashboard threshold breach: %s", name)

    return {
        "throughput": total,
        "recommendation_counts": recommendation_counts,
        "escalation_rate": round(escalation_count / total, 4) if total else None,
        "acceptance_rate": round(len(accepted) / len(decided), 4) if decided else None,
        "override_rate": round(len(overridden) / len(decided), 4) if decided else None,
        "decided_count": len(decided),
        "avg_cost_usd": avg_cost_usd,
        "max_cost_usd": round(max(costs), 6) if costs else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": p95_latency_ms,
        "cost_guardrail_usd": 0.08,
        "fairness_hard_block_pp": FAIRNESS_GAP_ALERT_PP,
        "fairness_gap_pp": fairness_gap_pp,
        "retrieval_failure_rate": retrieval_failure_rate,
        **alerts,
    }

from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import DecisionRecord, User

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def get_metrics(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Ops dashboard aggregate (US-407 subset): throughput, acceptance rate,
    override rate, avg cost, avg latency, current fairness gap status."""
    records = db.query(DecisionRecord).all()
    total = len(records)

    decided = [r for r in records if r.underwriter_action is not None]
    accepted = [r for r in decided if r.underwriter_action == "accept"]
    overridden = [r for r in decided if r.underwriter_action == "override"]

    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    latencies = [r.latency_ms for r in records if r.latency_ms is not None]

    recommendation_counts = dict(Counter(r.recommendation for r in records))

    escalation_count = sum(1 for r in records if r.escalation_flag)

    return {
        "throughput": total,
        "recommendation_counts": recommendation_counts,
        "escalation_rate": round(escalation_count / total, 4) if total else None,
        "acceptance_rate": round(len(accepted) / len(decided), 4) if decided else None,
        "override_rate": round(len(overridden) / len(decided), 4) if decided else None,
        "decided_count": len(decided),
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "max_cost_usd": round(max(costs), 6) if costs else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else None,
        "cost_guardrail_usd": 0.08,
        "fairness_hard_block_pp": 5.0,
    }

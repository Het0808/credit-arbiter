"""Decision-level fairness monitor with hard-block enforcement (FR-7 / US-304).

Distinct from the model-level audit in src/risk_model/fairness.py (which needs
ground-truth labels): this monitor runs over the *decision history*
(decision_record joined to application demographics) and computes, per loan
scheme, the approval-rate delta across demographic segments. When any segment
delta exceeds 5 percentage points, it raises a hard-block alert and PAUSES the
affected scheme (a SchemePause row) so the assessment service stops
auto-deciding that scheme until a human releases it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Application, DecisionRecord, SchemePause

HARD_BLOCK_DELTA_PP = 5.0  # percentage points
MIN_SUBGROUP_SIZE = 3      # ignore tiny subgroups to avoid noise


def _age_band(days_birth) -> str | None:
    if days_birth is None:
        return None
    years = abs(days_birth) / 365.25
    if years < 30:
        return "under_30"
    if years < 45:
        return "30_44"
    if years < 60:
        return "45_59"
    return "60_plus"


def _segment_value(application: Application, attribute: str):
    if attribute == "age_band":
        return _age_band(application.days_birth)
    return getattr(application, attribute, None)


SEGMENT_ATTRIBUTES = ["code_gender", "age_band", "name_education_type", "name_family_status", "region_rating_client"]


def compute_segment_deltas(rows: list[tuple[DecisionRecord, Application]]) -> dict:
    """Compute per-attribute approval-rate deltas for one scheme's decisions."""
    attribute_reports = {}
    max_delta = 0.0
    worst_attribute = None

    for attribute in SEGMENT_ATTRIBUTES:
        buckets: dict[str, list[int]] = {}
        for record, application in rows:
            val = _segment_value(application, attribute)
            if val is None:
                continue
            approved = int(record.recommendation == "Approve")
            buckets.setdefault(str(val), []).append(approved)

        subgroups = {
            k: {"n": len(v), "approval_rate": round(sum(v) / len(v), 4)}
            for k, v in buckets.items()
            if len(v) >= MIN_SUBGROUP_SIZE
        }
        if len(subgroups) >= 2:
            rates = [s["approval_rate"] for s in subgroups.values()]
            delta_pp = round((max(rates) - min(rates)) * 100, 2)
            attribute_reports[attribute] = {"subgroups": subgroups, "delta_pp": delta_pp}
            if delta_pp > max_delta:
                max_delta, worst_attribute = delta_pp, attribute

    return {"attributes": attribute_reports, "max_delta_pp": max_delta, "worst_attribute": worst_attribute}


def is_scheme_paused(db: Session, scheme: str) -> bool:
    return (
        db.query(SchemePause)
        .filter(SchemePause.scheme == scheme, SchemePause.released_at.is_(None))
        .first()
        is not None
    )


def paused_schemes(db: Session) -> list[SchemePause]:
    return db.query(SchemePause).filter(SchemePause.released_at.is_(None)).all()


def release_scheme(db: Session, scheme: str) -> int:
    active = db.query(SchemePause).filter(SchemePause.scheme == scheme, SchemePause.released_at.is_(None)).all()
    for pause in active:
        pause.released_at = datetime.utcnow()
    db.commit()
    return len(active)


def run_fairness_monitor(db: Session, enforce: bool = True) -> dict:
    """Evaluate every scheme's decision history and hard-block breaching schemes.

    When ``enforce`` is False the audit is read-only: it reports gaps/alerts but
    does not pause any scheme (used by the ops dashboard, which must not mutate
    state just by being viewed).
    """
    rows = (
        db.query(DecisionRecord, Application)
        .join(Application, DecisionRecord.application_id == Application.id)
        .all()
    )

    by_scheme: dict[str, list] = {}
    for record, application in rows:
        scheme = record.loan_scheme or application.loan_scheme or "Personal Loan"
        by_scheme.setdefault(scheme, []).append((record, application))

    scheme_reports, alerts, newly_paused = {}, [], []
    for scheme, scheme_rows in by_scheme.items():
        report = compute_segment_deltas(scheme_rows)
        report["decision_count"] = len(scheme_rows)
        breached = report["max_delta_pp"] > HARD_BLOCK_DELTA_PP
        report["hard_block"] = breached
        scheme_reports[scheme] = report

        if breached:
            reason = (
                f"Fairness gap {report['max_delta_pp']}pp on {report['worst_attribute']} "
                f"exceeds {HARD_BLOCK_DELTA_PP}pp"
            )
            alerts.append({"scheme": scheme, "reason": reason, "delta_pp": report["max_delta_pp"]})
            if enforce and not is_scheme_paused(db, scheme):
                db.add(SchemePause(scheme=scheme, reason=reason, gap_pp=report["max_delta_pp"]))
                newly_paused.append(scheme)

    if newly_paused:
        db.commit()

    return {
        "schemes": scheme_reports,
        "alerts": alerts,
        "newly_paused_schemes": newly_paused,
        "hard_block_threshold_pp": HARD_BLOCK_DELTA_PP,
        "healthy": not alerts,
    }

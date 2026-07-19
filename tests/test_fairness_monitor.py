"""Tests for the decision-level fairness monitor + hard-block (US-304)."""

import json

from src.api.models import Application, DecisionRecord
from src.api.services.fairness_monitor import (
    is_scheme_paused,
    release_scheme,
    run_fairness_monitor,
)


def _decision(db, *, gender, recommendation, scheme="Personal Loan"):
    app = Application(external_id=f"a{db.query(Application).count()}", loan_scheme=scheme,
                      code_gender=gender, days_birth=-14000, status="COMPLETE")
    db.add(app)
    db.commit()
    db.refresh(app)
    rec = DecisionRecord(
        application_id=app.id, loan_scheme=scheme, recommendation=recommendation,
        evidence_chain_json=json.dumps({}), escalation_flag=False,
    )
    db.add(rec)
    db.commit()


def test_no_gap_is_healthy(db_session):
    # Balanced approvals across genders -> no delta.
    for g in ("M", "F"):
        for _ in range(3):
            _decision(db_session, gender=g, recommendation="Approve")
            _decision(db_session, gender=g, recommendation="Refer")
    report = run_fairness_monitor(db_session)
    assert report["healthy"] is True
    assert report["newly_paused_schemes"] == []


def test_large_gap_triggers_hard_block_and_pause(db_session):
    # All males approved, all females referred -> 100pp gap.
    for _ in range(4):
        _decision(db_session, gender="M", recommendation="Approve")
        _decision(db_session, gender="F", recommendation="Refer")
    report = run_fairness_monitor(db_session)
    assert report["healthy"] is False
    assert "Personal Loan" in report["newly_paused_schemes"]
    assert is_scheme_paused(db_session, "Personal Loan") is True


def test_release_unpauses_scheme(db_session):
    for _ in range(4):
        _decision(db_session, gender="M", recommendation="Approve")
        _decision(db_session, gender="F", recommendation="Refer")
    run_fairness_monitor(db_session)
    assert is_scheme_paused(db_session, "Personal Loan") is True
    released = release_scheme(db_session, "Personal Loan")
    assert released >= 1
    assert is_scheme_paused(db_session, "Personal Loan") is False


def test_read_only_monitor_does_not_pause(db_session):
    # A gap that would trigger a hard-block must NOT pause when enforce=False
    # (the ops dashboard uses this path and must not mutate state).
    for _ in range(4):
        _decision(db_session, gender="M", recommendation="Approve")
        _decision(db_session, gender="F", recommendation="Refer")
    report = run_fairness_monitor(db_session, enforce=False)
    assert report["healthy"] is False  # gap still detected/alerted
    assert report["newly_paused_schemes"] == []
    assert is_scheme_paused(db_session, "Personal Loan") is False

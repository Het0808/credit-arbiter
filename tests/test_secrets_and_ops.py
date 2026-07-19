"""Tests for secrets/least-privilege (US-406) and ops endpoints (US-407, US-401)."""

import re
from pathlib import Path

import pytest

from src.api.models import Application
from src.api.settings import ScopeError, check_scope, get_secret

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# High-signal secret-literal patterns (AWS keys, private keys, provider tokens).
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
]


def test_no_secret_literals_in_source():
    offenders = []
    for path in SRC_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append((str(path), pattern.pattern))
    assert not offenders, f"Secret literals found: {offenders}"


def test_least_privilege_scope_enforced():
    check_scope("policy_retrieval", "policy:read")  # allowed
    with pytest.raises(ScopeError):
        check_scope("policy_retrieval", "regulatory:verify")  # denied


def test_get_secret_reads_env_not_literal(monkeypatch):
    monkeypatch.setenv("HALCYON_TEST_SECRET", "from-env")
    assert get_secret("HALCYON_TEST_SECRET") == "from-env"


# --- Ops API (US-407, US-401, US-405) ---

def _seed(client, scheme="Personal Loan"):
    session = client.SessionLocal()
    app = Application(external_id="ops-1", loan_scheme=scheme, amt_income_total=250000,
                      amt_credit=300000, amt_annuity=15000, days_employed=-3650, status="COMPLETE")
    session.add(app); session.commit(); session.refresh(app)
    app_id = app.id
    session.close()
    return app_id


def test_dashboard_endpoint_returns_kpis(client, auth_headers):
    app_id = _seed(client)
    client.post("/api/assess", json={"application_id": app_id}, headers=auth_headers)
    resp = client.get("/api/ops/dashboard", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "kpis" in body and "thresholds" in body
    for key in ("throughput_decisions", "cost_per_app_usd", "fairness_gap_pp"):
        assert key in body["kpis"]


def test_audit_verify_endpoint(client, auth_headers):
    app_id = _seed(client)
    client.post("/api/assess", json={"application_id": app_id}, headers=auth_headers)
    resp = client.get("/api/ops/audit/verify", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["intact"] is True


def test_kill_switch_endpoint_roundtrip(client, auth_headers):
    on = client.post("/api/ops/kill-switch", json={"active": True}, headers=auth_headers)
    assert on.status_code == 200
    status = client.get("/api/ops/kill-switch", headers=auth_headers)
    assert status.json()["active"] is True
    client.post("/api/ops/kill-switch", json={"active": False}, headers=auth_headers)


def test_explanation_endpoint(client, auth_headers):
    app_id = _seed(client)
    decision = client.post("/api/assess", json={"application_id": app_id}, headers=auth_headers).json()
    resp = client.get(f"/api/assessments/{decision['id']}/explanation", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "narrative" in body and body["claims"]

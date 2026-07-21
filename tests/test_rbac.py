"""Role-based access control tests: applicant isolation vs ops access."""

import json

APP = {"AMT_INCOME_TOTAL": "200000", "AMT_CREDIT": "300000", "AMT_ANNUITY": "15000", "DAYS_EMPLOYED": "-3000"}


def _ingest(client, headers, sk_id):
    return client.post("/api/applications/ingest", headers=headers, json={**APP, "SK_ID_CURR": sk_id})


def test_applicant_sees_only_their_own_applications(client, applicant_headers, auth_headers):
    mine = _ingest(client, applicant_headers, "700001").json()["id"]
    other = _ingest(client, auth_headers, "700002").json()["id"]  # ops-owned

    applicant_list = client.get("/api/applications", headers=applicant_headers).json()
    assert [a["id"] for a in applicant_list] == [mine]

    ops_list = client.get("/api/applications", headers=auth_headers).json()
    ops_ids = {a["id"] for a in ops_list}
    assert mine in ops_ids and other in ops_ids  # ops sees everything


def test_applicant_cannot_view_another_users_application(client, applicant_headers, auth_headers):
    other = _ingest(client, auth_headers, "700003").json()["id"]
    resp = client.get(f"/api/applications/{other}", headers=applicant_headers)
    assert resp.status_code == 403


def test_applicant_cannot_assess_or_decide(client, applicant_headers):
    app_id = _ingest(client, applicant_headers, "700004").json()["id"]
    assert client.post("/api/assess", headers=applicant_headers, json={"application_id": app_id}).status_code == 403


def test_applicant_blocked_from_all_ops_tools(client, applicant_headers):
    assert client.get("/api/ops/dashboard", headers=applicant_headers).status_code == 403
    assert client.get("/api/assessments/queue/human-review", headers=applicant_headers).status_code == 403
    assert client.post("/api/fairness/monitor", headers=applicant_headers).status_code == 403
    assert client.post("/api/policy/reindex", headers=applicant_headers, json={}).status_code == 403


def test_ops_can_assess_and_see_all(client, auth_headers, applicant_headers):
    app_id = _ingest(client, applicant_headers, "700005").json()["id"]  # applicant-owned
    assess = client.post("/api/assess", headers=auth_headers, json={"application_id": app_id})
    assert assess.status_code == 200  # ops can assess any application


def test_applicant_status_pending_then_reflects_decision(client, applicant_headers):
    app_id = _ingest(client, applicant_headers, "700006").json()["id"]

    my = client.get("/api/applications/my", headers=applicant_headers).json()
    assert len(my) == 1 and my[0]["decision_status"] == "Pending"

    # Simulate an accepted Approve decision on this application.
    from src.api.models import DecisionRecord

    s = client.SessionLocal()
    s.add(DecisionRecord(application_id=app_id, recommendation="Approve", evidence_chain_json="{}",
                         underwriter_action="accept"))
    s.commit()
    s.close()

    my2 = client.get("/api/applications/my", headers=applicant_headers).json()
    assert my2[0]["decision_status"] == "Approved"


def test_role_returned_by_users_me(client, applicant_headers, auth_headers):
    assert client.get("/api/users/me", headers=applicant_headers).json()["role"] == "applicant"
    assert client.get("/api/users/me", headers=auth_headers).json()["role"] == "underwriter"

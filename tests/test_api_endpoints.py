def test_unauthenticated_request_is_rejected(client):
    response = client.get("/api/applications")
    assert response.status_code == 401


def test_queue_shows_seeded_applications(client, auth_headers, seeded_application):
    response = client.get("/api/applications", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 1
    assert any(app["id"] == seeded_application.id for app in body)


def test_get_application_detail(client, auth_headers, seeded_application):
    response = client.get(f"/api/applications/{seeded_application.id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["external_id"] == "100001"


def test_get_application_detail_404_for_unknown_id(client, auth_headers):
    response = client.get("/api/applications/99999", headers=auth_headers)
    assert response.status_code == 404


def test_full_assess_decision_audit_flow(client, auth_headers, seeded_application):
    assess_response = client.post(
        "/api/assess", json={"application_id": seeded_application.id}, headers=auth_headers
    )
    assert assess_response.status_code == 200
    decision = assess_response.json()
    assert decision["recommendation"] in {"Approve", "Decline", "Refer"}
    assert "risk_score" in decision
    assert "retrieved_clause_id" in decision
    assessment_id = decision["id"]

    accept_response = client.post(
        f"/api/assessments/{assessment_id}/decision",
        json={"action": "accept"},
        headers=auth_headers,
    )
    assert accept_response.status_code == 200
    assert accept_response.json()["underwriter_action"] == "accept"

    audit_response = client.get(f"/api/assessments/{assessment_id}", headers=auth_headers)
    assert audit_response.status_code == 200
    audit = audit_response.json()
    assert audit["underwriter_action"] == "accept"
    assert audit["underwriter_action_at"] is not None


def test_override_requires_a_reason(client, auth_headers, seeded_application):
    assess_response = client.post(
        "/api/assess", json={"application_id": seeded_application.id}, headers=auth_headers
    )
    assessment_id = assess_response.json()["id"]

    missing_reason = client.post(
        f"/api/assessments/{assessment_id}/decision",
        json={"action": "override"},
        headers=auth_headers,
    )
    assert missing_reason.status_code == 400

    with_reason = client.post(
        f"/api/assessments/{assessment_id}/decision",
        json={"action": "override", "reason": "Manual income verification differs from stated income"},
        headers=auth_headers,
    )
    assert with_reason.status_code == 200
    assert with_reason.json()["underwriter_reason"]


def test_assess_404_for_unknown_application(client, auth_headers):
    response = client.post("/api/assess", json={"application_id": 99999}, headers=auth_headers)
    assert response.status_code == 404


def test_ingest_endpoint_flags_incomplete_without_crashing(client, auth_headers):
    response = client.post(
        "/api/applications/ingest",
        json={
            "SK_ID_CURR": "555001",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "AMT_INCOME_TOTAL": "100000",
            "AMT_CREDIT": "300000",
            "AMT_ANNUITY": "",
            "DAYS_EMPLOYED": "-400",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "INCOMPLETE"
    assert "AMT_ANNUITY" in body["missing_fields"]

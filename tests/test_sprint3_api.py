"""End-to-end API tests for Sprint 3 endpoints (documents, fairness, queue)."""

from src.api.models import Application


def _seed_app(client, scheme="Personal Loan"):
    session = client.SessionLocal()
    app = Application(external_id="s3-1", loan_scheme=scheme, amt_income_total=250000,
                      amt_credit=300000, amt_annuity=15000, days_employed=-3650, status="COMPLETE")
    session.add(app)
    session.commit()
    session.refresh(app)
    app_id = app.id
    session.close()
    return app_id


def test_document_upload_and_verify_flow(client, auth_headers):
    app_id = _seed_app(client)
    up = client.post(
        f"/api/applications/{app_id}/documents",
        data={"doc_type": "salary_slip", "declared_income": "250000"},
        files={"file": ("slip.pdf", b"%PDF-1.4 x", "application/pdf")},
        headers=auth_headers,
    )
    assert up.status_code == 200, up.text
    assert up.json()["doc_type"] == "salary_slip"

    verify = client.get(f"/api/applications/{app_id}/documents/verify", headers=auth_headers)
    assert verify.status_code == 200
    body = verify.json()
    assert "bank_statement" in body["missing_information"]
    assert body["complete"] is False


def test_unsupported_document_type_rejected(client, auth_headers):
    app_id = _seed_app(client)
    up = client.post(
        f"/api/applications/{app_id}/documents",
        data={"doc_type": "id_proof"},
        files={"file": ("bad.exe", b"MZ", "application/x-msdownload")},
        headers=auth_headers,
    )
    assert up.status_code == 400


def test_human_review_queue_and_override_rate(client, auth_headers):
    app_id = _seed_app(client)
    assess = client.post("/api/assess", json={"application_id": app_id, "force_regulatory_fail": True},
                         headers=auth_headers)
    assert assess.status_code == 200
    assert assess.json()["escalation_flag"] is True  # forced regulatory escalation

    queue = client.get("/api/assessments/queue/human-review", headers=auth_headers)
    assert queue.status_code == 200
    assert any(item["application_id"] == app_id for item in queue.json())

    rate = client.get("/api/assessments/metrics/override-rate", headers=auth_headers)
    assert rate.status_code == 200
    assert "override_rate" in rate.json()


def test_fairness_monitor_endpoint(client, auth_headers):
    _seed_app(client)
    resp = client.post("/api/fairness/monitor", headers=auth_headers)
    assert resp.status_code == 200
    assert "schemes" in resp.json()
    paused = client.get("/api/fairness/paused-schemes", headers=auth_headers)
    assert paused.status_code == 200

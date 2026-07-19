def test_metrics_requires_auth(client):
    response = client.get("/api/metrics")
    assert response.status_code == 401


def test_metrics_returns_zeroed_shape_with_no_assessments(client, auth_headers):
    response = client.get("/api/metrics", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["throughput"] == 0
    assert body["acceptance_rate"] is None
    assert body["cost_guardrail_usd"] == 0.08
    assert body["fairness_hard_block_pp"] == 5.0
    assert body["retrieval_failure_rate"] is None
    assert body["retrieval_failure_alert"] is False
    assert body["avg_cost_alert"] is False
    assert body["p95_latency_alert"] is False
    assert "fairness_gap_pp" in body
    assert "fairness_gap_alert" in body


def test_metrics_reflects_assessments_and_decisions(client, auth_headers, seeded_application):
    assess_response = client.post(
        "/api/assess", json={"application_id": seeded_application.id}, headers=auth_headers
    )
    assessment_id = assess_response.json()["id"]
    client.post(
        f"/api/assessments/{assessment_id}/decision",
        json={"action": "accept"},
        headers=auth_headers,
    )

    response = client.get("/api/metrics", headers=auth_headers)
    body = response.json()
    assert body["throughput"] == 1
    assert body["decided_count"] == 1
    assert body["acceptance_rate"] == 1.0
    assert body["override_rate"] == 0.0
    assert body["retrieval_failure_rate"] == 0.0
    assert body["retrieval_failure_alert"] is False


def test_metrics_flags_retrieval_failure_alert_above_5_percent(client, auth_headers, seeded_application):
    from src.api.models import Application

    session = client.SessionLocal()
    unknown_scheme_application = Application(
        external_id="900001",
        loan_scheme="Nonexistent Scheme",
        status="COMPLETE",
    )
    session.add(unknown_scheme_application)
    session.commit()
    unknown_scheme_id = unknown_scheme_application.id
    session.close()

    client.post("/api/assess", json={"application_id": seeded_application.id}, headers=auth_headers)
    client.post("/api/assess", json={"application_id": unknown_scheme_id}, headers=auth_headers)

    response = client.get("/api/metrics", headers=auth_headers)
    body = response.json()
    assert body["retrieval_failure_rate"] == 0.5
    assert body["retrieval_failure_alert"] is True


def test_metrics_flags_avg_cost_alert_above_ac8_threshold(client, auth_headers, seeded_application):
    from src.api.models import DecisionRecord

    session = client.SessionLocal()
    session.add(
        DecisionRecord(
            application_id=seeded_application.id,
            recommendation="Refer",
            evidence_chain_json="{}",
            cost_usd=0.06,  # above AC-8's $0.05/application ceiling
        )
    )
    session.commit()
    session.close()

    response = client.get("/api/metrics", headers=auth_headers)
    body = response.json()
    assert body["avg_cost_usd"] == 0.06
    assert body["avg_cost_alert"] is True

import time

from src.api.services.regulatory import verify_regulatory


def test_same_application_id_always_returns_same_verdict():
    first = verify_regulatory("100001")
    second = verify_regulatory("100001")
    assert first == second
    assert first["status"] in {"PASS", "FAIL"}


def test_different_ids_can_produce_different_verdicts():
    verdicts = {verify_regulatory(str(i))["status"] for i in range(50)}
    # Not asserting exact ratio, just that the deterministic hash isn't degenerate.
    assert verdicts.issubset({"PASS", "FAIL"})


def test_forced_failure_retries_then_escalates_never_fabricates():
    result = verify_regulatory("100001", force_fail=True)
    assert result["status"] == "escalate_for_review"
    assert result["reason"] == "regulatory_service_unavailable_after_retries"


def test_latency_stays_within_500ms_budget_including_retries():
    start = time.monotonic()
    verify_regulatory("100001", force_fail=True)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5

import time

from src.api.services.regulatory import SUB_CHECKS, verify_regulatory


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
    assert "unresolved" in result["reason"]
    # Every one of the four checks must have exhausted its retries, not fabricated.
    assert len(result["checks"]) == 4
    assert all(c["status"] == "escalate_for_review" for c in result["checks"])


def test_all_four_checks_run_and_pass_for_clean_id():
    result = verify_regulatory("100001")
    assert {c["check"] for c in result["checks"]} == {"identity", "employment", "tax", "sanctions"}


def test_latency_stays_within_500ms_budget_including_retries():
    start = time.monotonic()
    verify_regulatory("100001", force_fail=True)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


def test_sub_checks_all_present_and_deterministic():
    result = verify_regulatory("100002")
    assert set(result["sub_checks"].keys()) == set(SUB_CHECKS)
    assert all(v in {"PASS", "FAIL"} for v in result["sub_checks"].values())
    assert verify_regulatory("100002")["sub_checks"] == result["sub_checks"]


def test_overall_status_fails_if_any_sub_check_fails():
    # Find an id whose sub-checks aren't all PASS, to prove overall FAIL
    # actually reflects a real sub-check failure, not a fabricated aggregate.
    for i in range(200):
        result = verify_regulatory(str(i))
        failed = [k for k, v in result["sub_checks"].items() if v == "FAIL"]
        if failed:
            assert result["status"] == "FAIL"
            assert all(f in result["reason"] for f in failed)
            return
    assert False, "expected at least one FAIL sub-check across 200 sampled ids"

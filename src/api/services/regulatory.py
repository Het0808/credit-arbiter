"""Mock regulatory validation (FR-6 / US-109, US-303).

Four independently-checked sub-services (identity, employment, tax,
sanctions) - still intentionally mocks, per the PRD's declared v1 scope and
the out-of-scope declaration for live bureau/KYC integrations. Each sub-check
never fabricates a verdict: a simulated technical failure of the mock
external call is retried a bounded number of times and, if still failing,
surfaces escalate_for_review rather than guessing PASS or FAIL.
"""

import hashlib
import time

# Four checks run sequentially, each with up to MAX_RETRIES exponential-backoff
# retries; the base is kept small so the worst case (all checks down) still lands
# well inside the 500ms regulatory latency budget: 4 x (0.01+0.02+0.04) = 0.28s.
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.01

CHECKS = ("identity", "employment", "tax", "sanctions")


SUB_CHECKS = ["identity", "employment", "tax", "sanctions"]

# Sanctions hits should be rare in a mock population; the others mirror the
# original single-check 90% pass rate.
_PASS_THRESHOLD = {"identity": 90, "employment": 90, "tax": 90, "sanctions": 99}


def _sub_check_verdict(application_id: str, check_name: str) -> str:
    """Deterministic PASS/FAIL per (application, sub-check) pair, so repeated
    calls always return the same verdict for the same applicant."""
    digest = int(hashlib.sha256(f"{check_name}:{application_id}".encode("utf-8")).hexdigest(), 16)
    return "PASS" if digest % 100 < _PASS_THRESHOLD[check_name] else "FAIL"


def verify_regulatory(application_id: str, force_fail: bool = False) -> dict:
    """Return {"status": "PASS" | "FAIL" | "escalate_for_review", "reason": str | None,
    "sub_checks": {identity, employment, tax, sanctions} -> "PASS"/"FAIL"}.

    Overall status is PASS only if every sub-check passes. force_fail
    simulates a transient failure of the mock external regulatory service
    (distinct from a genuine FAIL business verdict): it is retried, then
    escalated - never fabricated.
    """
    if force_fail:
        for _ in range(MAX_RETRIES):
            time.sleep(RETRY_BACKOFF_SECONDS)
        return {
            "status": "escalate_for_review",
            "reason": "regulatory_service_unavailable_after_retries",
            "sub_checks": {},
        }

    sub_checks = {check: _sub_check_verdict(application_id, check) for check in SUB_CHECKS}
    failed = [name for name, verdict in sub_checks.items() if verdict == "FAIL"]
    status = "PASS" if not failed else "FAIL"
    reason = None if status == "PASS" else f"failed_checks:{','.join(failed)}"
    return {"status": status, "reason": reason, "sub_checks": sub_checks}

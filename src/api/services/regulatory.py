"""Mock regulatory validation stub (FR-6 / US-109).

This is intentionally a stub, per the PRD's declared Sprint 1 scope and the
out-of-scope declaration for live bureau/KYC integrations in v1. It never
fabricates a verdict: a simulated technical failure of the mock external call
is retried a bounded number of times and, if still failing, surfaces
escalate_for_review rather than guessing PASS or FAIL.
"""

import hashlib
import time

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.05


def _business_verdict(application_id: str) -> str:
    """Deterministic PASS/FAIL derived from a stable hash of the application id,
    so repeated calls for the same id always return the same verdict."""
    digest = int(hashlib.sha256(str(application_id).encode("utf-8")).hexdigest(), 16)
    return "PASS" if digest % 100 < 90 else "FAIL"


def verify_regulatory(application_id: str, force_fail: bool = False) -> dict:
    """Return {"status": "PASS" | "FAIL" | "escalate_for_review", "reason": str | None}.

    force_fail simulates a transient failure of the mock external regulatory
    service (distinct from a genuine FAIL business verdict): it is retried,
    then escalated - never fabricated.
    """
    if force_fail:
        for _ in range(MAX_RETRIES):
            time.sleep(RETRY_BACKOFF_SECONDS)
        return {"status": "escalate_for_review", "reason": "regulatory_service_unavailable_after_retries"}

    return {"status": _business_verdict(application_id), "reason": None}

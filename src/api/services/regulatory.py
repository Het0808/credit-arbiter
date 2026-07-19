"""Mock regulatory validation (FR-6 / US-303, hardening US-109).

Simulates four independent regulatory checks - identity, employment, tax, and
sanctions - each behind a mock external call that may transiently fail. Failing
calls are retried with exponential back-off; if a check cannot be resolved
after the configured retries it is surfaced as ``escalate_for_review`` rather
than a fabricated PASS/FAIL. The overall verdict is:

  - FAIL     if any check returns a business FAIL,
  - escalate_for_review if any check is unresolved after retries (and none FAIL),
  - PASS     only if all four checks resolve to PASS.

Deterministic per application id so repeated calls are stable and testable.
"""

import hashlib
import time

# Four checks run sequentially, each with up to MAX_RETRIES exponential-backoff
# retries; the base is kept small so the worst case (all checks down) still lands
# well inside the 500ms regulatory latency budget: 4 x (0.01+0.02+0.04) = 0.28s.
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.01

CHECKS = ("identity", "employment", "tax", "sanctions")


def _hash(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest(), 16)


def _check_verdict(application_id: str, check: str) -> str:
    """Deterministic PASS/FAIL for one check (sanctions fails rarely; others ~8%)."""
    digest = _hash(application_id, check)
    fail_rate = 3 if check == "sanctions" else 8
    return "FAIL" if digest % 100 < fail_rate else "PASS"


def _mock_external_call(application_id: str, check: str, force_fail: bool) -> str:
    """Return a verdict, or raise ConnectionError to simulate a transient outage."""
    if force_fail:
        raise ConnectionError(f"mock {check} service unavailable")
    return _check_verdict(application_id, check)


def _run_check_with_retries(application_id: str, check: str, force_fail: bool) -> dict:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            verdict = _mock_external_call(application_id, check, force_fail)
            return {"check": check, "status": verdict, "attempts": attempt + 1, "reason": None}
        except ConnectionError as exc:
            last_error = str(exc)
            time.sleep(BASE_BACKOFF_SECONDS * (2 ** attempt))  # exponential back-off
    return {
        "check": check,
        "status": "escalate_for_review",
        "attempts": MAX_RETRIES,
        "reason": f"unresolved_after_{MAX_RETRIES}_retries: {last_error}",
    }


def verify_regulatory(application_id: str, force_fail: bool = False) -> dict:
    """Run all four regulatory checks and return the aggregate verdict.

    Backwards-compatible shape: top-level ``status`` in {PASS, FAIL,
    escalate_for_review} and ``reason``, plus a per-check ``checks`` breakdown.
    ``force_fail`` simulates every external call being down (transient), which
    must escalate - never fabricate a verdict.
    """
    results = [_run_check_with_retries(application_id, c, force_fail) for c in CHECKS]

    failed = [r for r in results if r["status"] == "FAIL"]
    unresolved = [r for r in results if r["status"] == "escalate_for_review"]

    if failed:
        status = "FAIL"
        reason = "failed_checks: " + ", ".join(r["check"] for r in failed)
    elif unresolved:
        status = "escalate_for_review"
        reason = "unresolved_checks: " + ", ".join(r["check"] for r in unresolved)
    else:
        status = "PASS"
        reason = None

    return {"status": status, "reason": reason, "checks": results}

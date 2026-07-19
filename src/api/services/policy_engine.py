"""Multi-scheme policy evaluation engine (FR-3 / US-206).

Applies the machine-readable ``rule`` attached to each retrieved policy clause
to an application's normalised profile and returns which rules passed, which
failed, and whether human escalation is required. The engine is deliberately
deterministic and side-effect free so it is fully unit-testable and its output
is a defensible logical layer under every recommendation (AC-4).

Contract used by the assessment service (US-106 hardened):
- ``approve_allowed`` is False whenever any rule fails or cannot be evaluated,
  so a failed rule can never coexist with an Approve (policy adherence = 100%).
- ``required_action`` is the most severe remediation demanded by any failed
  rule ("decline" > "refer" > "escalate"), or None when all rules pass.
"""

from __future__ import annotations

SENTINEL_EMPLOYMENT = 0  # HC2018 uses a positive sentinel; any value >= 0 => no tenure
THIN_FILE_MONTHS = 24
DAYS_PER_MONTH = 30.44

# Severity ordering for choosing the single governing action when several rules fail.
_ACTION_SEVERITY = {"escalate": 1, "refer": 2, "decline": 3}

_OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def compute_metrics(profile: dict) -> dict:
    """Derive the policy metrics from a normalised applicant profile.

    Values that cannot be computed from the available inputs are returned as
    None; the evaluator treats a None metric as an unverifiable (failed) rule.
    Never reads code_gender or days_birth (assumption A-8b).
    """
    income = profile.get("amt_income_total")
    credit = profile.get("amt_credit")
    annuity = profile.get("amt_annuity")
    days_employed = profile.get("days_employed")

    def _ratio(numer, denom):
        if numer is None or not denom or denom <= 0:
            return None
        return numer / denom

    if days_employed is None:
        employment_months = None
    elif days_employed >= SENTINEL_EMPLOYMENT:
        employment_months = 0.0  # unemployed / pensioner sentinel
    else:
        employment_months = abs(days_employed) / DAYS_PER_MONTH

    return {
        "dti": _ratio(annuity, income),
        "lti": _ratio(credit, income),
        "annual_income": income,
        "employment_months": employment_months,
        "region_rating": profile.get("region_rating_client"),
        "thin_file": None if employment_months is None else employment_months < THIN_FILE_MONTHS,
    }


def _evaluate_clause(clause: dict, metrics: dict) -> dict | None:
    """Evaluate one clause's rule. Returns None for clauses with no rule
    (e.g. the fairness guardrail, which is informational, not auto-evaluable)."""
    rule = clause.get("rule")
    if not rule:
        return None

    metric_name = rule["metric"]
    observed = metrics.get(metric_name)
    op = _OPERATORS[rule["operator"]]
    threshold = rule["threshold"]
    on_violation = rule.get("on_violation", "refer")

    result = {
        "clause_id": clause["clause_id"],
        "scheme": clause.get("scheme"),
        "metric": metric_name,
        "operator": rule["operator"],
        "threshold": threshold,
        "observed": observed,
        "on_violation": on_violation,
    }

    # An optional guard gates whether the rule applies at all (e.g. thin-file
    # only escalates when LTI is also high). If the guard is not satisfied, the
    # rule is not applicable and counts as passed.
    guard = rule.get("guard")
    if guard is not None:
        guard_observed = metrics.get(guard["metric"])
        if guard_observed is None or not _OPERATORS[guard["operator"]](guard_observed, guard["threshold"]):
            result.update(satisfied=True, applicable=False, reason="guard_not_met")
            return result

    result["applicable"] = True

    if observed is None:
        # Cannot verify compliance -> treat as a failed rule needing at least referral.
        result.update(satisfied=False, reason="metric_unavailable")
        if on_violation == "escalate":
            result["on_violation"] = "refer"  # can't confirm thin-file -> refer, don't silently escalate
        return result

    satisfied = op(observed, threshold)
    result["satisfied"] = satisfied
    if not satisfied:
        result["reason"] = "threshold_violated"
    return result


def evaluate(clauses: list[dict], profile: dict, scheme: str = None, corpus_version: str = None) -> dict:
    """Evaluate a list of retrieved clauses against an application profile.

    Returns passed/failed rules, escalation requirements, the single governing
    ``required_action``, and ``approve_allowed`` (False if anything failed).
    """
    metrics = compute_metrics(profile)

    evaluated, passed, failed = [], [], []
    for clause in clauses:
        outcome = _evaluate_clause(clause, metrics)
        if outcome is None:
            continue
        evaluated.append(outcome)
        (passed if outcome["satisfied"] else failed).append(outcome)

    required_action = None
    if failed:
        required_action = max((f["on_violation"] for f in failed), key=lambda a: _ACTION_SEVERITY.get(a, 0))

    escalations = [f for f in failed if f["on_violation"] == "escalate"]

    return {
        "scheme": scheme,
        "corpus_version": corpus_version,
        "metrics": metrics,
        "evaluated_rules": evaluated,
        "passed_rules": [p["clause_id"] for p in passed],
        "failed_rules": [
            {"clause_id": f["clause_id"], "on_violation": f["on_violation"], "reason": f.get("reason")}
            for f in failed
        ],
        "escalation_required": bool(escalations),
        "escalation_reasons": [e["clause_id"] for e in escalations],
        "required_action": required_action,
        "approve_allowed": len(failed) == 0,
        "policy_adherence": round(len(passed) / len(evaluated), 4) if evaluated else 1.0,
    }

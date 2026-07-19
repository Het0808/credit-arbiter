"""Multi-scheme policy evaluation engine (FR-3 / US-206).

Applies machine-checkable numeric rules to an applicant profile, scheme-aware.
The thresholds here are deliberately kept in lockstep with the corresponding
clause text authored in data/policy_corpus_v1.0.json (POL-PL-001/002/003,
POL-ED-001/003, POL-VH-001/002, POL-BU-001/002, POL-FT-002/003, POL-LI-001/002)
so a retrieved clause and the programmatic check it grounds never disagree.

This is a real logical layer, not FR-4's retrieval - it runs independently of
which clause the RAG step happened to retrieve, so a policy violation is
still caught even if retrieval surfaces a different (but still relevant)
clause for the recommendation's citation.
"""

from typing import Any, Dict, Optional

# scheme -> {max_dti, min_income, max_lti, always_thin_file}. A None value
# means that scheme has no authored numeric rule for that dimension (kept
# unset rather than invented, so code and clause text never contradict).
SCHEME_RULES: Dict[str, Dict[str, Any]] = {
    "Personal": {"max_dti": 0.50, "min_income": 15000, "max_lti": 6.0, "clause_ids": ["POL-PL-001", "POL-PL-002", "POL-PL-003"]},
    "Education": {"max_dti": 0.45, "min_income": None, "max_lti": 8.0, "clause_ids": ["POL-ED-001", "POL-ED-003"]},
    "Vehicle": {"max_dti": None, "min_income": 12000, "max_lti": 5.0, "clause_ids": ["POL-VH-001", "POL-VH-002"]},
    "Business": {"max_dti": 0.55, "min_income": 50000, "max_lti": None, "clause_ids": ["POL-BU-001", "POL-BU-002"]},
    "First-Time Borrower": {"max_dti": None, "min_income": 12000, "max_lti": 4.0, "always_thin_file": True, "clause_ids": ["POL-FT-002", "POL-FT-003"]},
    "Low-Income Assistance": {"max_dti": 0.45, "min_income": 8000, "max_lti": None, "clause_ids": ["POL-LI-001", "POL-LI-002"]},
}

DEFAULT_SCHEME = "Personal"


def evaluate_policy(profile: dict, scheme: Optional[str]) -> Dict[str, Any]:
    """Evaluate an applicant profile against its scheme's numeric policy rules.

    Returns {scheme, passed_rules, failed_rules, thin_file_flag, escalation_required}.
    passed_rules/failed_rules are lists of {rule, clause_id, detail}.
    escalation_required is True whenever thin_file_flag is set or any rule
    failed (PRD §11: "Any policy rule violation or policy conflict" and
    "thin-file flag raised" are both mandatory escalation triggers).
    """
    scheme = scheme if scheme in SCHEME_RULES else DEFAULT_SCHEME
    rules = SCHEME_RULES[scheme]
    clause_ids = rules["clause_ids"]

    income = profile.get("amt_income_total")
    credit = profile.get("amt_credit")
    annuity = profile.get("amt_annuity")
    days_employed = profile.get("days_employed")

    passed_rules = []
    failed_rules = []

    dti = (annuity / income) if income and annuity is not None else None
    lti = (credit / income) if income and credit is not None else None

    if rules.get("max_dti") is not None and dti is not None:
        entry = {
            "rule": "max_dti",
            "clause_id": clause_ids[0],
            "detail": f"DTI {dti:.1%} vs cap {rules['max_dti']:.0%}",
        }
        (passed_rules if dti <= rules["max_dti"] else failed_rules).append(entry)

    if rules.get("min_income") is not None and income is not None:
        entry = {
            "rule": "min_income",
            "clause_id": clause_ids[0],
            "detail": f"income ${income:,.0f} vs floor ${rules['min_income']:,.0f}",
        }
        (passed_rules if income >= rules["min_income"] else failed_rules).append(entry)

    if rules.get("max_lti") is not None and lti is not None:
        entry = {
            "rule": "max_lti",
            "clause_id": clause_ids[-1],
            "detail": f"LTI {lti:.2f}x vs cap {rules['max_lti']:.1f}x",
        }
        (passed_rules if lti <= rules["max_lti"] else failed_rules).append(entry)

    thin_file_flag = bool(rules.get("always_thin_file")) or days_employed is None or days_employed >= 0
    escalation_required = thin_file_flag or bool(failed_rules)

    return {
        "scheme": scheme,
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
        "thin_file_flag": thin_file_flag,
        "escalation_required": escalation_required,
    }

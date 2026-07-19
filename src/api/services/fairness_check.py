"""Live fairness hard-block (FR-7 / US-304, AC-5).

Consults the precomputed reports/ml/fairness_thresholds.json (built from the
champion model's held-out-set fairness audit by
src.risk_model.fairness_thresholds, which must be re-run after every
retrain) for the applicant's demographic segments. Reading CODE_GENDER and
DAYS_BIRTH here is the sanctioned use assumption A-8b carves out for them
("retained exclusively for fairness monitoring") - the ML score itself never
sees them (see services/scoring.py).

If any applicable segment's approval-rate delta from the population baseline
exceeds the 5 percentage-point guardrail, this is a hard ship-block per
AC-5's literal language: the recommendation may not be Approve/Decline
without human review, regardless of risk score or policy outcome.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

_THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "reports" / "ml" / "fairness_thresholds.json"
_CACHE: Optional[dict] = None


def _load_thresholds() -> dict:
    global _CACHE
    if _CACHE is None:
        if _THRESHOLDS_PATH.exists():
            with open(_THRESHOLDS_PATH) as f:
                _CACHE = json.load(f)
        else:
            _CACHE = {"segments": {}}
    return _CACHE


def _age_band(days_birth: Optional[int]) -> Optional[str]:
    if days_birth is None:
        return None
    age_years = abs(days_birth) / 365.25
    for cutoff, label in [(25, "18-25"), (35, "26-35"), (45, "36-45"), (55, "46-55"), (65, "56-65")]:
        if age_years < cutoff:
            return label
    return "65+"


def check_fairness(profile: Dict[str, Any]) -> Dict[str, Any]:
    """profile must include code_gender, days_birth, name_education_type,
    name_income_type, region_rating_client - fairness-audit-only fields."""
    thresholds = _load_thresholds()
    segments = thresholds.get("segments", {})

    applicant_segments = {
        "CODE_GENDER": profile.get("code_gender"),
        "AGE_BAND": _age_band(profile.get("days_birth")),
        "NAME_EDUCATION_TYPE": profile.get("name_education_type"),
        "NAME_INCOME_TYPE": profile.get("name_income_type"),
        "REGION_RATING_CLIENT": (
            str(profile["region_rating_client"]) if profile.get("region_rating_client") is not None else None
        ),
    }

    triggered = []
    for attribute, subgroup in applicant_segments.items():
        if subgroup is None:
            continue
        seg_data = segments.get(attribute, {}).get(str(subgroup))
        if seg_data and seg_data.get("hard_block"):
            triggered.append(
                {
                    "attribute": attribute,
                    "subgroup": subgroup,
                    "delta_pp": seg_data["delta_pp"],
                }
            )

    return {"fairness_alert": bool(triggered), "triggered_segments": triggered}


def current_max_fairness_gap_pp() -> Optional[float]:
    """Largest |delta_pp| across all segments in the current fairness
    thresholds file (US-407 dashboard gauge) - the population-level gap
    from the champion model's last retrain audit, not a per-request value."""
    segments = _load_thresholds().get("segments", {})
    deltas = [abs(seg["delta_pp"]) for attr in segments.values() for seg in attr.values() if "delta_pp" in seg]
    return round(max(deltas), 2) if deltas else None

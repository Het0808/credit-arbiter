"""Application ingestion: CSV row -> normalised applicant profile (FR-1 / US-102).

This is the real ingestion/normalisation pipeline. Only the input file
(data/sample_applications.csv) is a synthetic stand-in for the real Home
Credit Default Risk 2018 dataset - see data/README.md. Swapping in the real
CSV later requires no changes to this module, only pointing scripts/seed_db.py
(or a future bulk-ingest job) at the real file.
"""

import csv
import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models import Application

# Exactly the fields the placeholder scorer (services/scoring.py) needs.
REQUIRED_FIELDS = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED"]

# HC2018 has no native loan-purpose/scheme field, so scheme assignment is a
# documented Sprint-2 simplification (US-203/204): applicants whose profile
# matches a specialty program's defining signal (income floor, no employment
# history) route there; everyone else is distributed deterministically across
# the remaining general-purpose schemes (same hash-based determinism style as
# the mock regulatory check in services/regulatory.py, not a real underwriting
# signal - real loan-purpose data would replace this at ingestion time).
_GENERAL_SCHEMES = ["Personal", "Education", "Vehicle", "Business"]
LOW_INCOME_ASSISTANCE_FLOOR = 15000


def _blank(value) -> bool:
    return value is None or str(value).strip() == ""


def _derive_loan_scheme(external_id: str, amt_income_total, days_employed) -> str:
    if not _blank(amt_income_total) and float(amt_income_total) < LOW_INCOME_ASSISTANCE_FLOOR:
        return "Low-Income Assistance"
    if not _blank(days_employed) and float(days_employed) >= 0:
        # HC2018's positive-sentinel (365243) or any non-negative value marks
        # unemployed/pensioner/no-bureau-record applicants - thin-file by the
        # same rule POL-PL-004 already uses.
        return "First-Time Borrower"
    digest = hashlib.sha256((external_id or "").encode()).hexdigest()
    return _GENERAL_SCHEMES[int(digest, 16) % len(_GENERAL_SCHEMES)]


def normalize_row(row: dict) -> dict:
    """Normalise one raw CSV row into the applicant-profile shape.

    Never raises: missing required fields are captured in missing_fields and
    status is set to INCOMPLETE instead of crashing ingestion.
    """
    missing = [field for field in REQUIRED_FIELDS if _blank(row.get(field))]

    profile = {
        "external_id": (row.get("SK_ID_CURR") or "").strip() or None,
        "name_contract_type": row.get("NAME_CONTRACT_TYPE") or None,
        "name_education_type": row.get("NAME_EDUCATION_TYPE") or None,
        "name_family_status": row.get("NAME_FAMILY_STATUS") or None,
        "occupation_type": row.get("OCCUPATION_TYPE") or None,
        "name_income_type": row.get("NAME_INCOME_TYPE") or None,
        "flag_own_car": row.get("FLAG_OWN_CAR") or None,
        "flag_own_realty": row.get("FLAG_OWN_REALTY") or None,
        # Stored for fairness-audit purposes only - never read by scoring.py (A-8b).
        "code_gender": row.get("CODE_GENDER") or None,
    }

    for field, key in [
        ("AMT_INCOME_TOTAL", "amt_income_total"),
        ("AMT_CREDIT", "amt_credit"),
        ("AMT_ANNUITY", "amt_annuity"),
        ("EXT_SOURCE_1", "ext_source_1"),
        ("EXT_SOURCE_2", "ext_source_2"),
        ("EXT_SOURCE_3", "ext_source_3"),
        ("CNT_FAM_MEMBERS", "cnt_fam_members"),
        ("AMT_GOODS_PRICE", "amt_goods_price"),
    ]:
        raw = row.get(field)
        profile[key] = float(raw) if not _blank(raw) else None

    for field, key in [
        ("DAYS_EMPLOYED", "days_employed"),
        ("DAYS_BIRTH", "days_birth"),
        ("REGION_RATING_CLIENT", "region_rating_client"),
        ("CNT_CHILDREN", "cnt_children"),
    ]:
        raw = row.get(field)
        profile[key] = int(float(raw)) if not _blank(raw) else None

    profile["status"] = "INCOMPLETE" if missing else "COMPLETE"
    profile["missing_fields"] = ",".join(missing) if missing else None
    profile["loan_scheme"] = _derive_loan_scheme(
        profile["external_id"], profile.get("amt_income_total"), profile.get("days_employed")
    )
    return profile


def load_csv_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ingest_row(db: Session, row: dict) -> Application:
    """Normalise a raw row and upsert it into the applications table by external_id."""
    profile = normalize_row(row)
    if not profile["external_id"]:
        raise ValueError("row is missing SK_ID_CURR / external_id, cannot ingest")

    existing: Optional[Application] = (
        db.query(Application).filter(Application.external_id == profile["external_id"]).first()
    )
    target = existing or Application(external_id=profile["external_id"])

    for key, value in profile.items():
        setattr(target, key, value)
    target.raw_row_json = json.dumps(row)

    if not existing:
        db.add(target)
    db.commit()
    db.refresh(target)
    return target

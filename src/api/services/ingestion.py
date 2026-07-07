"""Application ingestion: CSV row -> normalised applicant profile (FR-1 / US-102).

This is the real ingestion/normalisation pipeline. Only the input file
(data/sample_applications.csv) is a synthetic stand-in for the real Home
Credit Default Risk 2018 dataset - see data/README.md. Swapping in the real
CSV later requires no changes to this module, only pointing scripts/seed_db.py
(or a future bulk-ingest job) at the real file.
"""

import csv
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models import Application

# Exactly the fields the placeholder scorer (services/scoring.py) needs.
REQUIRED_FIELDS = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED"]


def _blank(value) -> bool:
    return value is None or str(value).strip() == ""


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
        # Stored for fairness-audit purposes only - never read by scoring.py (A-8b).
        "code_gender": row.get("CODE_GENDER") or None,
    }

    for field, key in [
        ("AMT_INCOME_TOTAL", "amt_income_total"),
        ("AMT_CREDIT", "amt_credit"),
        ("AMT_ANNUITY", "amt_annuity"),
    ]:
        raw = row.get(field)
        profile[key] = float(raw) if not _blank(raw) else None

    for field, key in [
        ("DAYS_EMPLOYED", "days_employed"),
        ("DAYS_BIRTH", "days_birth"),
        ("REGION_RATING_CLIENT", "region_rating_client"),
    ]:
        raw = row.get(field)
        profile[key] = int(float(raw)) if not _blank(raw) else None

    profile["status"] = "INCOMPLETE" if missing else "COMPLETE"
    profile["missing_fields"] = ",".join(missing) if missing else None
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

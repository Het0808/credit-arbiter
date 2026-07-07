"""Idempotent Sprint 1 POC seed script.

Run from the repo root (with the venv activated):

    python -m scripts.seed_db

Populates:
  - applications table from data/sample_applications.csv (upsert by external_id)
  - policy_corpus_version metadata row (upsert by version + source_file)
  - a demo underwriter login so the demo script doesn't require UI registration

Never seeds decision_record rows - those are only created by real /assess calls.
"""

import json
import os

from src.api.auth import get_password_hash
from src.api.database import Base, SessionLocal, engine
from src.api.models import PolicyCorpusVersion, User
from src.api.services.ingestion import ingest_row, load_csv_rows

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO_ROOT, "data", "sample_applications.csv")
POLICY_CORPUS_JSON = os.path.join(REPO_ROOT, "data", "policy_corpus_personal_loan_v0.1.json")

DEMO_USER_EMAIL = "underwriter@halcyon.com"
DEMO_USER_PASSWORD = "halcyon-demo-1"


def seed_policy_corpus(db) -> str:
    with open(POLICY_CORPUS_JSON, encoding="utf-8") as fh:
        corpus = json.load(fh)

    existing = (
        db.query(PolicyCorpusVersion)
        .filter(
            PolicyCorpusVersion.version == corpus["version"],
            PolicyCorpusVersion.source_file == POLICY_CORPUS_JSON,
        )
        .first()
    )
    if existing:
        return "skipped (already recorded)"

    db.add(
        PolicyCorpusVersion(
            version=corpus["version"],
            effective_date=corpus["effective_date"],
            source_file=POLICY_CORPUS_JSON,
            clause_count=len(corpus["clauses"]),
        )
    )
    db.commit()
    return "inserted"


def seed_demo_user(db) -> str:
    existing = db.query(User).filter(User.email == DEMO_USER_EMAIL).first()
    if existing:
        return "skipped (already exists)"

    db.add(
        User(
            email=DEMO_USER_EMAIL,
            hashed_password=get_password_hash(DEMO_USER_PASSWORD),
            role="underwriter",
        )
    )
    db.commit()
    return f"inserted (login: {DEMO_USER_EMAIL} / {DEMO_USER_PASSWORD})"


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        rows = load_csv_rows(SAMPLE_CSV)
        for row in rows:
            ingest_row(db, row)
        print(f"applications: upserted {len(rows)} rows from {SAMPLE_CSV}")

        corpus_result = seed_policy_corpus(db)
        print(f"policy_corpus_version: {corpus_result}")

        user_result = seed_demo_user(db)
        print(f"demo underwriter user: {user_result}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

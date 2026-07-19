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
from src.api.services import retrieval as retrieval_service

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO_ROOT, "data", "sample_applications.csv")

DEMO_USER_EMAIL = "underwriter@halcyon.com"
DEMO_USER_PASSWORD = "halcyon-demo-1"


def seed_policy_corpus(db) -> str:
    """Record a metadata row for every discovered corpus version (US-207).

    The active version served at runtime is governed by the retrieval service
    (retrieval_service.active_version()); these rows exist for audit/replay.
    """
    inserted = 0
    for version in retrieval_service.list_versions()["available_versions"]:
        meta = retrieval_service.get_index(version).metadata()
        source_file = os.path.join(REPO_ROOT, "data", meta["source_file"])
        existing = (
            db.query(PolicyCorpusVersion)
            .filter(
                PolicyCorpusVersion.version == version,
                PolicyCorpusVersion.source_file == source_file,
            )
            .first()
        )
        if existing:
            continue
        db.add(
            PolicyCorpusVersion(
                version=version,
                effective_date=meta["effective_date"],
                source_file=source_file,
                clause_count=meta["clause_count"],
            )
        )
        inserted += 1
    db.commit()
    active = retrieval_service.active_version()
    return f"{inserted} version(s) inserted; active={active}"


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

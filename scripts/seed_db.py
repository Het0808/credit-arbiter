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
from pathlib import Path

from src.api.auth import get_password_hash
from src.api.database import Base, SessionLocal, engine
from src.api.models import Application, Document, PolicyCorpusVersion, User
from src.api.services.document_verification import REQUIRED_DOCS_BY_SCHEME
from src.api.services.ingestion import ingest_row, load_csv_rows

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO_ROOT, "data", "sample_applications.csv")
POLICY_CORPUS_JSON = os.path.join(REPO_ROOT, "data", "policy_corpus_v1.0.json")
UPLOAD_ROOT = Path(REPO_ROOT) / "uploads"

DEMO_USER_EMAIL = "underwriter@halcyon.com"
DEMO_USER_PASSWORD = "halcyon-demo-1"

# Deliberately left without any uploaded documents, same demo intent as the
# CSV rows with missing required fields: shows the missing_documents
# escalation path actually firing, not just theoretically implemented.
DEMO_APPLICATION_WITHOUT_DOCS = "100004"


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


def seed_demo_documents(db) -> str:
    """Upload each COMPLETE application's scheme-required documents as
    placeholder files, except DEMO_APPLICATION_WITHOUT_DOCS - so the demo
    shows the full Approve/Decline/Refer range instead of every assessment
    escalating on missing documents once US-301/302 is wired into /assess."""
    if db.query(Document).count() > 0:
        return "skipped (already seeded)"

    count = 0
    for application in db.query(Application).filter(Application.status == "COMPLETE").all():
        if application.external_id == DEMO_APPLICATION_WITHOUT_DOCS:
            continue
        required = REQUIRED_DOCS_BY_SCHEME.get(application.loan_scheme, REQUIRED_DOCS_BY_SCHEME["Personal"])
        app_dir = UPLOAD_ROOT / str(application.id)
        app_dir.mkdir(parents=True, exist_ok=True)
        for doc_type in required:
            filename = f"{doc_type}.pdf"
            storage_path = app_dir / f"{doc_type}_{filename}"
            storage_path.write_bytes(b"%PDF-1.4 placeholder demo document\n")
            db.add(
                Document(
                    application_id=application.id,
                    doc_type=doc_type,
                    filename=filename,
                    storage_path=str(storage_path),
                )
            )
            count += 1
    db.commit()
    return f"inserted {count} placeholder documents"


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

        doc_result = seed_demo_documents(db)
        print(f"demo documents: {doc_result}")

        user_result = seed_demo_user(db)
        print(f"demo underwriter user: {user_result}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

"""Tests for document upload + verification (US-301, US-302)."""

from pathlib import Path

from src.api.models import Application
from src.api.services.document_service import (
    UnsupportedDocumentType,
    store_document,
    verify_documents,
)


def _app(db, scheme="Personal Loan", income=250000.0):
    application = Application(external_id="doc-1", loan_scheme=scheme, amt_income_total=income, status="COMPLETE")
    db.add(application)
    db.commit()
    db.refresh(application)
    return application


def test_supported_upload_is_stored_and_linked(db_session, tmp_path):
    app = _app(db_session)
    doc = store_document(
        db_session, app, doc_type="salary_slip", filename="slip.pdf",
        content_type="application/pdf", content=b"%PDF-1.4 fake",
    )
    assert doc.id is not None
    assert doc.application_id == app.id
    assert doc.size_bytes == len(b"%PDF-1.4 fake")


def test_unsupported_type_is_rejected(db_session):
    app = _app(db_session)
    try:
        store_document(db_session, app, doc_type="salary_slip", filename="evil.exe",
                       content_type="application/x-msdownload", content=b"MZ")
        assert False, "expected UnsupportedDocumentType"
    except UnsupportedDocumentType:
        pass


def test_verification_reports_missing_documents(db_session):
    app = _app(db_session)  # Personal Loan requires salary_slip, bank_statement, id_proof
    store_document(db_session, app, doc_type="salary_slip", filename="s.pdf",
                   content_type="application/pdf", content=b"x")
    report = verify_documents(db_session, app)
    assert report["complete"] is False
    assert set(report["missing_information"]) == {"bank_statement", "id_proof"}


def test_verification_flags_income_inconsistency(db_session):
    app = _app(db_session, income=250000.0)
    for dtype, income in [("salary_slip", 250000.0), ("bank_statement", 250000.0), ("id_proof", 100000.0)]:
        store_document(db_session, app, doc_type=dtype, filename=f"{dtype}.pdf",
                       content_type="application/pdf", content=b"x", declared_income=income)
    report = verify_documents(db_session, app)
    assert report["complete"] is True
    assert any(f["type"] == "income_mismatch_across_docs" for f in report["consistency_findings"])
    assert report["verified"] is False


def test_path_traversal_filename_is_sanitized(db_session):
    app = _app(db_session)
    doc = store_document(
        db_session, app, doc_type="id_proof", filename="../../../../tmp/evil.txt",
        content_type="text/plain", content=b"x",
    )
    # Stored strictly under data/uploads/<app_id>/, using only the base name.
    from src.api.services.document_service import UPLOAD_DIR
    resolved = Path(doc.storage_path).resolve()
    assert str(resolved).startswith(str((UPLOAD_DIR / str(app.id)).resolve()))
    assert resolved.name == "evil.txt"


def test_verification_passes_when_complete_and_consistent(db_session):
    app = _app(db_session, income=250000.0)
    for dtype in ("salary_slip", "bank_statement", "id_proof"):
        store_document(db_session, app, doc_type=dtype, filename=f"{dtype}.pdf",
                       content_type="application/pdf", content=b"x",
                       declared_name="Jane Doe", declared_income=250000.0)
    report = verify_documents(db_session, app)
    assert report["verified"] is True

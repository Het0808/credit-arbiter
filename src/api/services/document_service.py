"""Document storage (FR-5 / US-301) and verification (US-302).

Storage accepts a bounded set of file types, persists the bytes under
data/uploads/<application_id>/, and links a Document row to the application.
Verification runs two checks against the scheme's required-document list:
  - completeness: which required document types are missing, and
  - consistency: whether declared name / income agree across documents (and
    with the application's recorded income) - a basic fraud/typo signal.

Real OCR is out of scope for the POC, so declared_name / declared_income are
supplied as structured metadata at upload time; the checks are identical to
what they would be over OCR-extracted fields.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.orm import Session

from ..models import Application, Document

REPO_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"

# Supported upload content types (US-301 AC).
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "text/plain",
}
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}

# Required document types per loan scheme (US-302). Falls back to the personal
# loan set for any scheme not explicitly listed.
REQUIRED_DOCS = {
    "Personal Loan": ["salary_slip", "bank_statement", "id_proof"],
    "Education Loan": ["salary_slip", "bank_statement", "id_proof", "admission_letter"],
    "Vehicle Loan": ["salary_slip", "bank_statement", "id_proof"],
    "Business Loan": ["bank_statement", "id_proof", "business_registration"],
    "First-Time Buyer Loan": ["salary_slip", "bank_statement", "id_proof", "address_proof"],
    "Low-Income Loan": ["bank_statement", "id_proof"],
}
DEFAULT_REQUIRED = REQUIRED_DOCS["Personal Loan"]

INCOME_TOLERANCE = 0.15  # 15% relative difference tolerated before flagging


class UnsupportedDocumentType(ValueError):
    """Raised when an uploaded file's type is not supported (US-301)."""


def required_docs_for_scheme(scheme: str | None) -> list[str]:
    return REQUIRED_DOCS.get(scheme or "", DEFAULT_REQUIRED)


def store_document(
    db: Session,
    application: Application,
    *,
    doc_type: str,
    filename: str,
    content_type: str | None,
    content: bytes,
    declared_name: str | None = None,
    declared_income: float | None = None,
) -> Document:
    """Validate the file type, persist bytes to disk, and create a Document row."""
    ext = os.path.splitext(filename)[1].lower()
    if content_type not in SUPPORTED_CONTENT_TYPES and ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentType(
            f"Unsupported document type: {content_type or ext!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Strip any directory components from the client-supplied filename to prevent
    # path traversal (e.g. '../../etc/passwd'); keep only the base name.
    safe_name = os.path.basename(filename) or "upload"
    app_dir = UPLOAD_DIR / str(application.id)
    app_dir.mkdir(parents=True, exist_ok=True)
    storage_path = app_dir / safe_name
    storage_path.write_bytes(content)

    document = Document(
        application_id=application.id,
        doc_type=doc_type,
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        storage_path=str(storage_path),
        declared_name=declared_name,
        declared_income=declared_income,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def verify_documents(db: Session, application: Application) -> dict:
    """Run completeness + consistency checks for an application's documents (US-302)."""
    scheme = application.loan_scheme or "Personal Loan"
    required = required_docs_for_scheme(scheme)
    documents = db.query(Document).filter(Document.application_id == application.id).all()
    present_types = {d.doc_type for d in documents}

    missing = [d for d in required if d not in present_types]

    # Consistency: declared names should agree; declared incomes should agree
    # with each other and with the application's recorded income.
    consistency_findings = []

    names = {d.declared_name.strip().lower() for d in documents if d.declared_name}
    if len(names) > 1:
        consistency_findings.append(
            {"type": "name_mismatch", "detail": f"Conflicting names across documents: {sorted(names)}"}
        )

    incomes = [d.declared_income for d in documents if d.declared_income is not None]
    if incomes:
        lo, hi = min(incomes), max(incomes)
        if lo > 0 and (hi - lo) / lo > INCOME_TOLERANCE:
            consistency_findings.append(
                {"type": "income_mismatch_across_docs", "detail": f"Declared incomes range {lo:.0f}-{hi:.0f}"}
            )
        if application.amt_income_total:
            declared_mean = sum(incomes) / len(incomes)
            app_income = application.amt_income_total
            if abs(declared_mean - app_income) / app_income > INCOME_TOLERANCE:
                consistency_findings.append(
                    {
                        "type": "income_mismatch_vs_application",
                        "detail": f"Docs avg {declared_mean:.0f} vs application {app_income:.0f}",
                    }
                )

    complete = not missing
    return {
        "scheme": scheme,
        "required_docs": required,
        "present_docs": sorted(present_types),
        "missing_information": missing,
        "complete": complete,
        "consistency_findings": consistency_findings,
        "consistent": not consistency_findings,
        "verified": complete and not consistency_findings,
        "document_count": len(documents),
    }

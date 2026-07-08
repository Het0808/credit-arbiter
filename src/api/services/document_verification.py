"""Document verification (FR-5 / US-301, US-302).

Completeness checking (required docs present per scheme) is real logic run
against whatever has actually been uploaded and stored. Cross-document
consistency checking (e.g. name/income agreeing across documents) would
require real OCR/document parsing, which is out of scope for v1 (PRD §5) -
so, like the existing mock regulatory check (services/regulatory.py), it is
a deterministic, clearly-labelled mock that never fabricates a specific
finding, following the same hash-based determinism pattern already used
elsewhere in this codebase.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

ALLOWED_DOC_TYPES = {
    "salary_slip",
    "bank_statement",
    "employment_letter",
    "id_proof",
    "address_proof",
    "explanation_letter",
    "enrollment_letter",
    "insurance_proof",
    "revenue_statement",
    "income_proof",
}

# Required doc types per loan scheme - a documented Sprint-3 business rule,
# not derived from any dataset (HC2018 has no document metadata).
REQUIRED_DOCS_BY_SCHEME: Dict[str, List[str]] = {
    "Personal": ["salary_slip", "bank_statement", "id_proof"],
    "Education": ["enrollment_letter", "id_proof"],
    "Vehicle": ["id_proof", "insurance_proof"],
    "Business": ["revenue_statement", "id_proof"],
    "First-Time Borrower": ["id_proof", "address_proof"],
    "Low-Income Assistance": ["id_proof", "income_proof"],
}


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_supported_doc_type(doc_type: str) -> bool:
    return doc_type in ALLOWED_DOC_TYPES


def check_completeness(scheme: str, uploaded_doc_types: List[str]) -> Dict[str, Any]:
    """Compare uploaded document types against the scheme's required list."""
    required = REQUIRED_DOCS_BY_SCHEME.get(scheme, REQUIRED_DOCS_BY_SCHEME["Personal"])
    uploaded_set = set(uploaded_doc_types)
    missing = [doc for doc in required if doc not in uploaded_set]
    return {"required": required, "missing_documents": missing, "complete": not missing}


def check_consistency(application_external_id: str) -> Dict[str, Any]:
    """Mocked cross-document consistency check (no real OCR/parsing in v1).

    Deterministic per applicant (same hash-based-determinism pattern as
    services/regulatory.py's mock verdict), never fabricates a specific
    finding: either "no discrepancy found" or a generic named finding type.
    """
    digest = hashlib.sha256(f"consistency:{application_external_id}".encode()).hexdigest()
    consistent = (int(digest, 16) % 100) < 92  # ~92% of applicants show no discrepancy
    findings = [] if consistent else ["name_or_income_mismatch_across_documents"]
    return {"consistent": consistent, "findings": findings}


def verify_documents(scheme: str, application_external_id: str, uploaded_doc_types: List[str]) -> Dict[str, Any]:
    completeness = check_completeness(scheme, uploaded_doc_types)
    consistency = check_consistency(application_external_id)
    return {
        "required_documents": completeness["required"],
        "missing_documents": completeness["missing_documents"],
        "complete": completeness["complete"],
        "consistent": consistency["consistent"],
        "consistency_findings": consistency["findings"],
    }

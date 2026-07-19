from src.api.services.document_verification import (
    check_completeness,
    check_consistency,
    is_supported_doc_type,
    is_supported_file,
    verify_documents,
)


def test_complete_uploads_pass():
    result = check_completeness("Personal", ["salary_slip", "bank_statement", "id_proof"])
    assert result["complete"] is True
    assert result["missing_documents"] == []


def test_missing_required_doc_is_flagged():
    result = check_completeness("Personal", ["salary_slip"])
    assert result["complete"] is False
    assert "bank_statement" in result["missing_documents"]
    assert "id_proof" in result["missing_documents"]


def test_unknown_scheme_falls_back_to_personal_requirements():
    result = check_completeness("Nonexistent Scheme", [])
    assert result["required"] == ["salary_slip", "bank_statement", "id_proof"]


def test_consistency_check_is_deterministic_per_applicant():
    first = check_consistency("100001")
    second = check_consistency("100001")
    assert first == second


def test_supported_file_types():
    assert is_supported_file("payslip.pdf") is True
    assert is_supported_file("payslip.exe") is False


def test_supported_doc_types():
    assert is_supported_doc_type("salary_slip") is True
    assert is_supported_doc_type("not_a_real_doc_type") is False


def test_verify_documents_combines_completeness_and_consistency():
    result = verify_documents("Education", "100001", ["enrollment_letter", "id_proof"])
    assert result["complete"] is True
    assert "consistent" in result
    assert "consistency_findings" in result

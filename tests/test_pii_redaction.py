from src.api.services.pii_redaction import contains_pii, redact_pii, sanitize_for_llm


def test_ssn_is_redacted():
    text = "Applicant SSN is 123-45-6789 on file."
    redacted = redact_pii(text)
    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted


def test_account_number_is_redacted():
    text = "Account number 987654321 was verified."
    redacted = redact_pii(text)
    assert "987654321" not in redacted


def test_dob_is_redacted():
    for text in ["DOB: 04/12/1990", "DOB: 1990-04-12"]:
        redacted = redact_pii(text)
        assert "1990" not in redacted


def test_clean_text_has_no_pii():
    text = "Risk band Medium, DTI 42%, EXT_SOURCE_MEAN 0.55"
    assert contains_pii(text) is False
    assert redact_pii(text) == text


def test_sanitize_for_llm_blocks_when_pii_survives(monkeypatch):
    import src.api.services.pii_redaction as pii_module

    # Simulate a pattern gap: force contains_pii to report a leak even after
    # redaction, to exercise the fail-closed "blocked" path.
    monkeypatch.setattr(pii_module, "contains_pii", lambda text: True)
    sanitized, blocked = sanitize_for_llm("some evidence text")
    assert blocked is True


def test_sanitize_for_llm_passes_clean_text():
    sanitized, blocked = sanitize_for_llm("Risk band Low, DTI 12%")
    assert blocked is False
    assert sanitized == "Risk band Low, DTI 12%"

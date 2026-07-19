"""Unit tests for the PII redaction layer (US-209)."""

import pytest

from src.api.services.pii_redaction import PIILeakError, find_pii, redact, sanitize_for_llm


def test_redacts_ssn_dob_and_account():
    text = "Applicant SSN 123-45-6789, DOB 1985-03-12, account 12345678901."
    redacted, findings = redact(text)
    assert "123-45-6789" not in redacted
    assert "1985-03-12" not in redacted
    assert "12345678901" not in redacted
    types = {f["type"] for f in findings}
    assert {"SSN", "DOB", "ACCOUNT_NUMBER"}.issubset(types)


def test_sanitize_returns_pii_free_prompt():
    prompt = "Contact jane@bank.com about SSN 111-22-3333."
    clean = sanitize_for_llm(prompt, context="unit_test")
    assert find_pii(clean) == []
    assert "111-22-3333" not in clean
    assert "jane@bank.com" not in clean


def test_clean_text_passes_through_unchanged():
    prompt = "Applicant has a debt-to-income ratio of 0.42 and requests a personal loan."
    assert sanitize_for_llm(prompt) == prompt


def test_leak_is_blocked(monkeypatch):
    # Force redact() to leave PII behind so the re-lint gate must block the call.
    from src.api.services import pii_redaction as mod

    monkeypatch.setattr(mod, "redact", lambda text: (text, []))
    with pytest.raises(PIILeakError):
        sanitize_for_llm("SSN 123-45-6789 leaks through")

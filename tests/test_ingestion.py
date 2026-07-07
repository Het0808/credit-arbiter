from src.api.services.ingestion import normalize_row


def _base_row(**overrides):
    row = {
        "SK_ID_CURR": "100001",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "AMT_INCOME_TOTAL": "250000",
        "AMT_CREDIT": "300000",
        "AMT_ANNUITY": "15000",
        "DAYS_EMPLOYED": "-3650",
        "DAYS_BIRTH": "-14200",
        "CODE_GENDER": "F",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "REGION_RATING_CLIENT": "1",
        "OCCUPATION_TYPE": "Core staff",
    }
    row.update(overrides)
    return row


def test_valid_row_is_complete():
    profile = normalize_row(_base_row())
    assert profile["status"] == "COMPLETE"
    assert profile["missing_fields"] is None
    assert profile["external_id"] == "100001"
    assert profile["amt_income_total"] == 250000.0
    assert profile["amt_credit"] == 300000.0
    assert profile["amt_annuity"] == 15000.0
    assert profile["days_employed"] == -3650


def test_missing_required_field_flags_incomplete_without_crashing():
    profile = normalize_row(_base_row(AMT_ANNUITY=""))
    assert profile["status"] == "INCOMPLETE"
    assert "AMT_ANNUITY" in profile["missing_fields"]
    assert profile["amt_annuity"] is None


def test_multiple_missing_fields_are_all_captured():
    profile = normalize_row(_base_row(AMT_ANNUITY="", AMT_INCOME_TOTAL=""))
    assert profile["status"] == "INCOMPLETE"
    assert "AMT_ANNUITY" in profile["missing_fields"]
    assert "AMT_INCOME_TOTAL" in profile["missing_fields"]


def test_days_employed_sentinel_handled_without_crashing():
    profile = normalize_row(_base_row(DAYS_EMPLOYED="365243"))
    assert profile["status"] == "COMPLETE"
    assert profile["days_employed"] == 365243


def test_missing_external_id_is_reported_as_missing_field():
    profile = normalize_row(_base_row(SK_ID_CURR=""))
    assert profile["external_id"] is None
    assert profile["status"] == "INCOMPLETE"
    assert "SK_ID_CURR" in profile["missing_fields"]

from src.api.services.fairness_check import _age_band, check_fairness, current_max_fairness_gap_pp


def test_age_band_buckets_correctly():
    assert _age_band(-365 * 20) == "18-25"
    assert _age_band(-365 * 40) == "36-45"
    assert _age_band(-365 * 70) == "65+"
    assert _age_band(None) is None


def test_young_applicant_triggers_fairness_alert():
    profile = {
        "code_gender": "F",
        "days_birth": -365 * 20,  # 20 years old -> 18-25 band, known hard_block segment
        "name_education_type": None,
        "name_income_type": None,
        "region_rating_client": None,
    }
    result = check_fairness(profile)
    assert result["fairness_alert"] is True
    assert any(seg["attribute"] == "AGE_BAND" for seg in result["triggered_segments"])


def test_middle_aged_applicant_does_not_trigger_alert():
    profile = {
        "code_gender": "F",
        "days_birth": -365 * 40,  # 40 years old -> 36-45 band, not hard_block
        "name_education_type": None,
        "name_income_type": None,
        "region_rating_client": None,
    }
    result = check_fairness(profile)
    assert result["fairness_alert"] is False


def test_missing_segment_data_never_crashes():
    result = check_fairness({})
    assert result["fairness_alert"] is False
    assert result["triggered_segments"] == []


def test_current_max_fairness_gap_pp_reads_real_thresholds_file():
    gap = current_max_fairness_gap_pp()
    assert gap is None or gap >= 0

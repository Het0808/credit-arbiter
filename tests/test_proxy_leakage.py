"""Tests for the proxy-leakage association measures (US-305)."""

import numpy as np
import pandas as pd

from src.risk_model.proxy_leakage import correlation_ratio, cramers_v


def test_cramers_v_detects_strong_association():
    # Perfectly aligned categoricals -> V near 1.
    a = pd.Series(["x", "x", "y", "y"] * 25)
    b = pd.Series(["m", "m", "f", "f"] * 25)
    assert cramers_v(a, b) > 0.8


def test_cramers_v_near_zero_for_independent():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(["x", "y"], size=400))
    b = pd.Series(rng.choice(["m", "f"], size=400))
    assert cramers_v(a, b) < 0.2


def test_correlation_ratio_detects_group_separation():
    cats = pd.Series(["young"] * 50 + ["old"] * 50)
    vals = pd.Series([25.0] * 50 + [65.0] * 50)
    assert correlation_ratio(cats, vals) > 0.9


def test_correlation_ratio_low_when_no_separation():
    rng = np.random.default_rng(1)
    cats = pd.Series(rng.choice(["a", "b"], size=200))
    vals = pd.Series(rng.normal(40, 5, size=200))
    assert correlation_ratio(cats, vals) < 0.3

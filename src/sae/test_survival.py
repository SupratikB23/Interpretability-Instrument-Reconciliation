"""Unit tests for the SAE survival taxonomy (2606.03002 §2.2)."""

import numpy as np

from src.sae.survival import classify_survival, survival_rates


def test_taxonomy_boundaries():
    cs = np.array([0.9001, 0.9, 0.5, 0.4999, 1.0, 0.7])
    labels = classify_survival(cs, survived_corr_min=0.9, degraded_corr_min=0.5)
    assert labels[0] == "survived"  # c > 0.9
    assert labels[1] == "degraded"  # c = 0.9 (0.5 <= c <= 0.9)
    assert labels[2] == "degraded"  # c = 0.5
    assert labels[3] == "damaged"   # c < 0.5
    assert labels[4] == "survived"
    assert labels[5] == "degraded"


def test_rates_over_active_only():
    # 4 features; firing rates: two active (f>0.001), two inactive.
    firing = np.array([0.01, 0.01, 0.0, 0.0])
    corr = np.array([0.95, 0.6, 0.1, 0.2])  # active: survived, degraded
    rates = survival_rates(corr, firing, 0.001, 0.9, 0.5)
    assert rates["survived"] == 50.0 and rates["degraded"] == 50.0 and rates["damaged"] == 0.0
    assert rates["survived"] + rates["degraded"] + rates["damaged"] == 100.0
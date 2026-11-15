"""
Automated checks for the scorecard module and the Streamlit app.

Run from the project folder with:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.credit_model import (  # noqa: E402
    EXCLUDED, FEATURES, TARGET, cost_optimal_threshold, ks_statistic, pd_to_score, psi, train_bundle,
)

APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def bundle():
    return train_bundle(str(ROOT / "data"))


# ---------------------------------------------------------------- model ----
def test_matches_notebook_results(bundle):
    """The dashboard model must reproduce the notebook's test-set numbers."""
    t = bundle.test
    assert len(t) == 6000
    assert roc_auc_score(t[TARGET], t["pd"]) == pytest.approx(0.7438, abs=1e-4)
    assert ks_statistic(t[TARGET], t["pd"]) == pytest.approx(0.3855, abs=1e-4)
    assert bundle.coefficients.round(3).tolist() == [-0.454, -0.599, -0.489, -0.367]


def test_protected_attributes_not_used(bundle):
    assert not set(EXCLUDED) & set(FEATURES)
    assert list(bundle.X_train.columns) == FEATURES


def test_pd_is_calibrated_overall(bundle):
    assert bundle.test["pd"].mean() == pytest.approx(bundle.test[TARGET].mean(), abs=0.01)


def test_all_coefficients_negative(bundle):
    """Higher WoE = safer, so every coefficient must be negative."""
    assert (bundle.coefficients < 0).all()


def test_points_add_up_to_score(bundle):
    for i in range(20):
        row = bundle.X_test.iloc[[i]]
        assert bundle.points_breakdown(row)["points"].sum() == pytest.approx(bundle.test["score"].iloc[i], abs=1e-6)


def test_cost_threshold_and_psi(bundle):
    assert cost_optimal_threshold(5, 1) == pytest.approx(1 / 6)
    assert psi(pd_to_score(bundle.train_pd), bundle.test["score"].values)["psi"].sum() < 0.01


# ------------------------------------------------------------------ app ----
def run_app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_without_errors():
    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 3
    assert metric(at, "Test AUC") == "0.744"


def test_late_account_scores_worse_and_gets_flagged():
    at = run_app()
    pd_before = float(metric(at, "Probability of default").rstrip("%"))
    at.selectbox[0].set_value("2 months late").run()
    assert not at.exception
    pd_after = float(metric(at, "Probability of default").rstrip("%"))
    assert pd_after > pd_before
    assert any("Flag for early intervention" in e.value for e in at.error)


def test_every_status_and_extreme_inputs():
    at = run_app()
    for status in at.selectbox[0].options:
        at.selectbox[0].set_value(status).run()
        assert not at.exception, status
    util = next(s for s in at.slider if s.label.startswith("Average utilization"))
    util.set_value(120).run()
    assert not at.exception


def test_cost_ratio_and_stress_sliders():
    at = run_app()
    for ratio in (1, 20):
        at.sidebar.slider[0].set_value(ratio).run()
        assert not at.exception
    slip = next(s for s in at.slider if s.label.startswith("Share of current customers"))
    slip.set_value(50).run()
    assert not at.exception
    assert metric(at, "Status") in {"Investigate", "Significant shift"}

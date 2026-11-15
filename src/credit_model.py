"""
credit_model.py
===============
Reusable model code for the Streamlit dashboard (app.py).

It reproduces, step for step, the final model built in
notebooks/credit_default_logistic_regression.ipynb:

    1. Load the data and engineer behavioural features
    2. Leave out protected attributes (sex, marital status, age) and education
    3. Split 80/20, stratified on the target (same seed as the notebook)
    4. Bin each feature and encode it with weight of evidence (WoE),
       learned on the TRAINING data only
    5. Fit logistic regression on the WoE values -> probability of default (PD)
    6. Convert PD into scorecard points (600 points = 50:1 odds, 20 points to double the odds)

Keeping this logic in one module means the notebook and the dashboard can't
drift apart: the automated tests check that the numbers match.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# Constants (identical to the notebook)
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "default_next_month"
DATA_FILE = "credit_card_default.csv"
DATA_URL = ("https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/"
            "main/dataset/default_of_credit_card_clients.csv")

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]      # repayment status, Sep -> Apr
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]                      # statement balance, Sep -> Apr
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]                    # amount paid, Sep -> Apr

# Not used by the model: protected grounds under the Canadian Human Rights Act
# (sex, marital status, age), plus education (undocumented codes, and a likely proxy).
EXCLUDED = ["SEX", "MARRIAGE", "AGE", "EDUCATION"]

# Payment ratio was a candidate too, but added nothing once repayment history was in
# the model (notebook section 6), so the final scorecard uses four features.
FEATURES = ["limit_bal", "pay_status_latest", "months_delinquent_6m", "avg_utilization_6m"]

FEATURE_LABELS = {
    "limit_bal": "Credit limit",
    "pay_status_latest": "Latest repayment status",
    "months_delinquent_6m": "Months delinquent (last 6)",
    "avg_utilization_6m": "Average utilization (6 months)",
}

# Bin edges: training-set quantiles (notebook section 5), rounded for readability.
# Each bin is (lower, upper]; -inf/inf make the outer bins open-ended.
BIN_EDGES = {
    "limit_bal": [-np.inf, 50_000, 100_000, 180_000, 270_000, np.inf],
    # -2, -1 and 0 are different behaviours (no use / paid in full / revolving), not degrees of
    # lateness, so each keeps its own bin; then 1, 2, and 3+ months late
    "pay_status_latest": [-np.inf, -2, -1, 0, 1, 2, np.inf],
    "months_delinquent_6m": [-np.inf, 0, 1, 2, np.inf],        # 0, 1, 2, 3+
    "avg_utilization_6m": [-np.inf, 0.02, 0.13, 0.45, 0.75, np.inf],
}
BIN_LABELS = {
    "limit_bal": ["≤ NT$50K", "NT$50K–100K", "NT$100K–180K", "NT$180K–270K", "> NT$270K"],
    "pay_status_latest": ["No card use", "Paid in full", "Revolving (paid minimum)",
                          "1 month late", "2 months late", "3+ months late"],
    "months_delinquent_6m": ["0", "1", "2", "3 or more"],
    "avg_utilization_6m": ["≤ 2% (inactive)", "2%–13%", "13%–45%", "45%–75%", "> 75%"],
}

# Scorecard scaling: 600 points at odds of 50 good : 1 bad; +20 points doubles the odds
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)                          # 28.85 points per unit of log-odds
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)  # 487.12


# --------------------------------------------------------------------------
# 1. Data loading and feature engineering
# --------------------------------------------------------------------------
def load_raw(data_dir: str) -> pd.DataFrame:
    """Read the dataset from data_dir, downloading it first if it's missing."""
    path = os.path.join(data_dir, DATA_FILE)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        tmp = os.path.join(data_dir, "download.csv")
        urllib.request.urlretrieve(DATA_URL, tmp)
        pd.read_csv(tmp).rename(columns={"default payment next month": TARGET}).to_csv(path, index=False)
        os.remove(tmp)
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn 6 months of raw account history into behavioural features
    (the 4 used by the scorecard, plus the payment ratio that was tested and dropped)."""
    f = pd.DataFrame(index=df.index)
    f["limit_bal"] = df["LIMIT_BAL"]          # credit limit, in New Taiwan dollars (NT$)
    # PAY_0 = September status: -2/-1 = paid in full or no use, 0 = revolving, 1..8 = months late
    f["pay_status_latest"] = df["PAY_0"]
    # Number of the last 6 months with any payment delay
    f["months_delinquent_6m"] = (df[PAY_COLS] >= 1).sum(axis=1)
    # Average balance across 6 statements, as a share of the credit limit
    f["avg_utilization_6m"] = df[BILL_COLS].mean(axis=1) / df["LIMIT_BAL"]
    # Payments made Sep..May divided by the bills they were paying (Aug..Apr statements).
    # PAY_AMT1 (paid in Sep) pays BILL_AMT2 (August statement), and so on.
    owed = df[BILL_COLS[1:]].clip(lower=0).sum(axis=1)
    paid = df[PAY_AMT_COLS[:5]].sum(axis=1)
    ratio = paid / owed.replace(0, np.nan)
    f["payment_ratio_5m"] = ratio.fillna(1.0).clip(0, 2)   # nothing owed -> treat as fully paid
    return f


# --------------------------------------------------------------------------
# 2. Binning and weight of evidence (WoE)
# --------------------------------------------------------------------------
def assign_bins(feature: str, values: pd.Series) -> pd.Series:
    """Map raw values to their bin index (0, 1, 2, ...) using BIN_EDGES."""
    return pd.cut(values, bins=BIN_EDGES[feature], labels=False, right=True).astype(int)


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Replace each feature's value with the weight of evidence of its bin.

    For bin i:   WoE_i = ln( share of all GOOD accounts in bin i / share of all BAD accounts in bin i )
    Positive WoE = safer than average, negative = riskier. 0.5 is added to each
    count (smoothing) so a bin with no defaults can't produce ln(0).

    fit() learns the WoE tables from the training data; inside a cross-validation
    Pipeline it is re-learned on each training fold, so no target information leaks.
    """

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.tables_ = {}
        for feat in FEATURES:
            bins = assign_bins(feat, X[feat]).values
            rows = []
            n_bins = len(BIN_EDGES[feat]) - 1
            total_bad, total_good = y.sum(), (1 - y).sum()
            for b in range(n_bins):
                in_bin = bins == b
                bad, good = y[in_bin].sum(), (1 - y[in_bin]).sum()
                pct_good = (good + 0.5) / (total_good + 0.5 * n_bins)
                pct_bad = (bad + 0.5) / (total_bad + 0.5 * n_bins)
                woe = np.log(pct_good / pct_bad)
                rows.append({"bin": b, "label": BIN_LABELS[feat][b], "accounts": int(in_bin.sum()),
                             "defaults": int(bad), "default_rate": bad / max(in_bin.sum(), 1),
                             "pct_good": pct_good, "pct_bad": pct_bad, "woe": woe,
                             "iv": (pct_good - pct_bad) * woe})
            self.tables_[feat] = pd.DataFrame(rows)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for feat in FEATURES:
            woe_by_bin = self.tables_[feat]["woe"].values
            out[feat] = woe_by_bin[assign_bins(feat, X[feat]).values]
        return out

    def information_value(self) -> pd.Series:
        """IV = sum over bins of (pct_good - pct_bad) x WoE: the feature's predictive strength."""
        return pd.Series({f: t["iv"].sum() for f, t in self.tables_.items()}).sort_values(ascending=False)


def make_scorecard_pipeline() -> Pipeline:
    """WoE encoding, then logistic regression: the selected model.
    C=1e6 makes the penalty negligible, i.e. ordinary (unpenalised) logistic regression."""
    return Pipeline([("woe", WoEEncoder()),
                     ("model", LogisticRegression(C=1e6, max_iter=1000))])


# --------------------------------------------------------------------------
# 3. Evaluation metrics, written out step by step
# --------------------------------------------------------------------------
def ks_statistic(y, pd_hat) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative share of bads
    and the cumulative share of goods, when accounts are sorted from riskiest to safest."""
    order = np.argsort(-np.asarray(pd_hat))
    y_sorted = np.asarray(y)[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    return float(np.max(cum_bad - cum_good))


def psi(expected_scores, actual_scores, n_bins: int = 10) -> pd.DataFrame:
    """Population stability index table.

    Bins are the deciles of the EXPECTED (training) scores. For each bin:
        PSI_i = (actual% - expected%) x ln(actual% / expected%)
    Total PSI < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift.
    """
    edges = np.unique(np.quantile(expected_scores, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.Series(pd.cut(expected_scores, edges)).value_counts(sort=False) / len(expected_scores)
    a = pd.Series(pd.cut(actual_scores, edges)).value_counts(sort=False).reindex(e.index) / len(actual_scores)
    e, a = e.clip(lower=1e-6), a.clip(lower=1e-6)
    t = pd.DataFrame({"expected_pct": e.values, "actual_pct": a.values}, index=e.index.astype(str))
    t["psi"] = (t["actual_pct"] - t["expected_pct"]) * np.log(t["actual_pct"] / t["expected_pct"])
    return t


def decile_table(y, pd_hat) -> pd.DataFrame:
    """Accounts split into 10 equal groups from riskiest to safest: predicted vs actual default rate."""
    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_hat)})
    d["decile"] = pd.qcut(d["pd"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(accounts=("y", "size"), defaults=("y", "sum"),
                                               avg_predicted_pd=("pd", "mean"), actual_default_rate=("y", "mean"))
    t["cum_share_of_defaults"] = t["defaults"].cumsum() / t["defaults"].sum()
    return t


# Risk bands by probability of default
RISK_BANDS = [("A", 0.00, 0.10, "Low"), ("B", 0.10, 0.15, "Moderate"), ("C", 0.15, 0.25, "Elevated"),
              ("D", 0.25, 0.50, "High"), ("E", 0.50, 1.01, "Very high")]


def risk_band(pd_value: float) -> tuple[str, str]:
    for band, lo, hi, name in RISK_BANDS:
        if lo <= pd_value < hi:
            return band, name
    return "E", "Very high"


def cost_optimal_threshold(cost_missed_default: float, cost_false_alarm: float) -> float:
    """With a calibrated PD, flag an account when the expected cost of NOT flagging it is higher:
        PD x cost_missed  >  (1 - PD) x cost_false_alarm
        ->  PD  >  cost_false_alarm / (cost_false_alarm + cost_missed)"""
    return cost_false_alarm / (cost_false_alarm + cost_missed_default)


# --------------------------------------------------------------------------
# 4. The trained model bundle
# --------------------------------------------------------------------------
@dataclass
class ScorecardBundle:
    pipeline: Pipeline
    train: pd.DataFrame            # raw training rows (incl. target and excluded columns)
    test: pd.DataFrame             # raw test rows, with pd and score columns added
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    train_pd: np.ndarray
    scorecard: pd.DataFrame        # points per feature bin

    @property
    def woe(self) -> WoEEncoder:
        return self.pipeline.named_steps["woe"]

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.pipeline.named_steps["model"].coef_[0], index=FEATURES)

    @property
    def intercept(self) -> float:
        return float(self.pipeline.named_steps["model"].intercept_[0])

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[FEATURES])[:, 1]

    def points_breakdown(self, X_row: pd.DataFrame) -> pd.DataFrame:
        """The scorecard points one account earns for each feature (they add up to its score)."""
        rows = []
        for feat in FEATURES:
            b = int(assign_bins(feat, X_row[feat]).iloc[0])
            sc = self.scorecard[(self.scorecard["feature"] == feat) & (self.scorecard["bin"] == b)].iloc[0]
            rows.append({"feature": FEATURE_LABELS[feat], "bin": sc["label"], "woe": sc["woe"], "points": sc["points"]})
        return pd.DataFrame(rows)


def pd_to_score(pd_hat) -> np.ndarray:
    """Score = OFFSET + FACTOR x ln(odds of good) , where odds of good = (1 - PD) / PD."""
    pd_hat = np.clip(np.asarray(pd_hat, float), 1e-9, 1 - 1e-9)
    return OFFSET + FACTOR * np.log((1 - pd_hat) / pd_hat)


def build_scorecard(pipeline: Pipeline) -> pd.DataFrame:
    """Points for every bin of every feature.

    log-odds of BAD = intercept + sum_j(beta_j x WoE_j), so
    score = OFFSET - FACTOR x (intercept + sum_j beta_j x WoE_j)
          = sum_j [ -FACTOR x beta_j x WoE_j  +  (OFFSET - FACTOR x intercept) / n ]
    Each bracket is the points for one feature, so the points add up to the score.
    """
    woe = pipeline.named_steps["woe"]
    model = pipeline.named_steps["model"]
    beta, alpha, n = model.coef_[0], model.intercept_[0], len(FEATURES)
    rows = []
    for j, feat in enumerate(FEATURES):
        for _, r in woe.tables_[feat].iterrows():
            pts = -FACTOR * beta[j] * r["woe"] + (OFFSET - FACTOR * alpha) / n
            rows.append({"feature": feat, "bin": int(r["bin"]), "label": r["label"],
                         "woe": r["woe"], "points": pts})
    return pd.DataFrame(rows)


def train_bundle(data_dir: str) -> ScorecardBundle:
    """Run the full notebook workflow and return a trained, evaluated scorecard."""
    raw = load_raw(data_dir)
    feats = engineer_features(raw)
    data = pd.concat([raw, feats], axis=1)

    # Stratified split keeps the 22% default rate identical in train and test
    train, test = train_test_split(data, test_size=0.2, stratify=data[TARGET], random_state=RANDOM_STATE)
    X_train, X_test = train[FEATURES], test[FEATURES]

    pipeline = make_scorecard_pipeline().fit(X_train, train[TARGET])
    test = test.copy()
    test["pd"] = pipeline.predict_proba(X_test)[:, 1]
    test["score"] = pd_to_score(test["pd"])
    train_pd = pipeline.predict_proba(X_train)[:, 1]

    return ScorecardBundle(pipeline=pipeline, train=train, test=test, X_train=X_train, X_test=X_test,
                           train_pd=train_pd, scorecard=build_scorecard(pipeline))

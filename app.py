"""
Credit Default Scorecard Dashboard
==================================
An interactive Streamlit dashboard on top of the logistic regression scorecard
built in notebooks/credit_default_logistic_regression.ipynb.

Run it locally with:
    streamlit run app.py

Three tabs:
    1. Score an account   : enter an account's history, get its PD, score, risk band,
                            points breakdown and the main reasons behind the score
    2. Cut-off strategy   : choose how many accounts to flag, and see what it catches and costs
    3. Model health       : discrimination, calibration, challengers, fairness and drift
                            (the model ownership view)

Streamlit re-runs this script top to bottom every time an input changes. Slow steps
(training the model and the challengers) are cached so they run only once.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from src.credit_model import (  # noqa: E402
    BIN_LABELS, FACTOR, FEATURE_LABELS, FEATURES, OFFSET, PAY_AMT_COLS, BILL_COLS, PAY_COLS,
    RANDOM_STATE, RISK_BANDS, TARGET, assign_bins, cost_optimal_threshold, decile_table,
    ks_statistic, make_scorecard_pipeline, pd_to_score, psi, risk_band, train_bundle,
)

st.set_page_config(page_title="Credit Default Scorecard", page_icon="💳", layout="wide")

NAVY, AMBER, RED, GREEN, GREY = "#2C3E50", "#E8A33D", "#C0392B", "#2E8B57", "#95A5A6"
STATUS_OPTIONS = {   # label -> PAY_0 code
    "No card use": -2, "Paid in full": -1, "Revolving (paid minimum)": 0,
    "1 month late": 1, "2 months late": 2, "3+ months late": 3,
}


# --------------------------------------------------------------------------
# Cached training (runs once per server start)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the scorecard (first load only)...")
def load_model():
    return train_bundle(str(APP_DIR / "data"))


@st.cache_resource(show_spinner="Training challenger models (first load only)...")
def load_challengers(_b):
    """Cross-validated and test AUC for the scorecard and two challengers (as in notebook section 13)."""
    y_tr, y_te = _b.train[TARGET], _b.test[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    raw_cols = ["LIMIT_BAL"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
    rows = []
    sc_cv = cross_val_score(make_scorecard_pipeline(), _b.X_train, y_tr, cv=cv, scoring="roc_auc").mean()
    rows.append(("WoE scorecard (selected)", 4, sc_cv, roc_auc_score(y_te, _b.test["pd"]), "Yes"))
    plain = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    pl_cv = cross_val_score(plain, _b.X_train, y_tr, cv=cv, scoring="roc_auc").mean()
    plain.fit(_b.X_train, y_tr)
    rows.append(("Plain logistic regression", 4, pl_cv, roc_auc_score(y_te, plain.predict_proba(_b.X_test)[:, 1]), "Partly"))
    gbm = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_cv = cross_val_score(gbm, _b.train[raw_cols], y_tr, cv=cv, scoring="roc_auc").mean()
    gbm.fit(_b.train[raw_cols], y_tr)
    rows.append(("Gradient boosting", len(raw_cols), gb_cv, roc_auc_score(y_te, gbm.predict_proba(_b.test[raw_cols])[:, 1]), "No"))
    t = pd.DataFrame(rows, columns=["Model", "Features", "CV AUC", "Test AUC", "Explainable as points"])
    t["Test Gini"] = 2 * t["Test AUC"] - 1
    return t, sc_cv


b = load_model()
test = b.test
y_test = test[TARGET].values
pd_test = test["pd"].values
train_scores = pd_to_score(b.train_pd)


def pct(x, d=1):
    return f"{x:.{d}%}"


def nt(x):
    return f"NT${x:,.0f}"


# --------------------------------------------------------------------------
# Sidebar: business cost assumption (drives the cut-off everywhere)
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Cut-off settings")
    cost_ratio = st.slider(
        "Cost of a missed default vs. a false alarm", 1, 20, 5, format="%d×",
        help="How many times more costly it is to miss a defaulter than to flag a good customer unnecessarily.")
    optimal_threshold = cost_optimal_threshold(cost_ratio, 1)
    st.markdown(f"**Cost-optimal cut-off:** flag if PD ≥ **{pct(optimal_threshold)}**  \n"
                f"= 1 ÷ (1 + {cost_ratio}) (score {pd_to_score(optimal_threshold):.0f})")
    st.divider()
    st.subheader("About")
    st.markdown(
        "Logistic regression scorecard predicting whether a credit card holder defaults **next month**, "
        "trained on 30,000 accounts (Taiwan, 2005; amounts in NT$).\n\n"
        "**Use case:** early warning, flagging accounts for a reminder, payment plan or limit review "
        "before the default happens.\n\n"
        "Sex, marital status, age and education are **not used** by the model.")

st.title("💳 Credit Default Scorecard")
st.caption("Probability of default (PD) model for early intervention: a weight-of-evidence logistic regression "
           "scorecard, its cut-off strategy and its health checks.")

tab_score, tab_cutoff, tab_health = st.tabs(["🧮 Score an account", "🎯 Cut-off strategy", "🩺 Model health"])


# ==========================================================================
# TAB 1: SCORE AN ACCOUNT
# ==========================================================================
with tab_score:
    st.subheader("Score one account")
    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        limit = st.select_slider("Credit limit", options=list(range(10_000, 1_000_001, 10_000)),
                                 value=140_000, format_func=nt)
        status_label = st.selectbox("Latest repayment status (this month)", list(STATUS_OPTIONS), index=2)
        status = STATUS_OPTIONS[status_label]
        # If the account is late this month, at least 1 of the last 6 months was delinquent
        min_months = 1 if status >= 1 else 0
        months_del = st.slider("Months with a late payment (last 6)", min_months, 6, max(min_months, 0),
                               help="Includes this month if it's late.")
        util = st.slider("Average utilization over 6 months", 0, 120, 29, format="%d%%",
                         help="Average statement balance as a share of the credit limit.") / 100

    row = pd.DataFrame([{"limit_bal": limit, "pay_status_latest": status,
                         "months_delinquent_6m": months_del, "avg_utilization_6m": util}])
    p = float(b.predict_pd(row)[0])
    score = float(pd_to_score(p))
    band, band_name = risk_band(p)
    breakdown = b.points_breakdown(row)

    # Reason codes: points lost versus the best bin of each feature
    best_points = b.scorecard.groupby("feature")["points"].max()
    breakdown["best_possible"] = [best_points[f] for f in FEATURES]
    breakdown["points_lost"] = breakdown["best_possible"] - breakdown["points"]

    with col_out:
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of default", pct(p), help="Chance this account defaults next month.")
        m2.metric("Score", f"{score:.0f}", help="Higher is safer. +20 points = twice the odds of paying.")
        m3.metric("Risk band", f"{band} · {band_name}")

        if p >= optimal_threshold:
            st.error(f"**Flag for early intervention.** PD {pct(p)} is at or above the cut-off of {pct(optimal_threshold)} "
                     f"(missed default costs {cost_ratio}× a false alarm).")
        else:
            st.success(f"**No action needed.** PD {pct(p)} is below the cut-off of {pct(optimal_threshold)}.")

        reasons = breakdown[breakdown["points_lost"] > 0.5].sort_values("points_lost", ascending=False).head(2)
        if len(reasons):
            st.markdown("**Main reasons the score isn't higher**")
            for _, r in reasons.iterrows():
                st.markdown(f"- **{r['feature']}: {r['bin']}**, {r['points_lost']:.0f} points below the best category")
        else:
            st.markdown("**Every feature is in its best category.**")

        # Points chart: earned vs best possible, per feature
        fig = go.Figure()
        fig.add_bar(y=breakdown["feature"], x=breakdown["points"], orientation="h", name="Points earned",
                    marker_color=NAVY, text=[f"{v:.0f}" for v in breakdown["points"]], textposition="inside")
        fig.add_bar(y=breakdown["feature"], x=breakdown["points_lost"], orientation="h", name="Points lost vs best category",
                    marker_color=AMBER, text=[f"−{v:.0f}" if v > 0.5 else "" for v in breakdown["points_lost"]],
                    textposition="inside")
        fig.update_layout(barmode="stack", height=300, margin=dict(t=40, b=10, l=10, r=10),
                          title=f"Scorecard points: total {score:.0f}", legend=dict(orientation="h", y=-0.15),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

        with st.expander("Show the calculation step by step"):
            w = b.woe.transform(row)
            logit = b.intercept + float((b.coefficients * w.iloc[0]).sum())
            st.markdown("**1. Bin each feature and look up its weight of evidence (WoE), learned on the training data**")
            st.dataframe(pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in FEATURES],
                "Bin": breakdown["bin"].values,
                "WoE": w.iloc[0].values,
                "Coefficient": b.coefficients.values,
                "Coefficient × WoE": (b.coefficients * w.iloc[0]).values,
                "Points": breakdown["points"].values,
            }).style.format({"WoE": "{:.3f}", "Coefficient": "{:.3f}", "Coefficient × WoE": "{:.3f}", "Points": "{:.1f}"}),
                hide_index=True, width="stretch")
            st.markdown(f"""
**2. Log-odds of default** = intercept + Σ(coefficient × WoE) = {b.intercept:.3f} + ({logit - b.intercept:.3f}) = **{logit:.3f}**

**3. Probability of default** = 1 ÷ (1 + e^−(log-odds)) = 1 ÷ (1 + e^{-logit:.3f}) = **{pct(p, 2)}**

**4. Score** = {OFFSET:.1f} + {FACTOR:.2f} × ln((1 − PD) ÷ PD) = {OFFSET:.1f} + {FACTOR:.2f} × ln({1 - p:.4f} ÷ {p:.4f}) = **{score:.1f}**,
which equals the sum of the points: {' + '.join(f'{v:.1f}' for v in breakdown['points'])} = {breakdown['points'].sum():.1f}

**5. Decision:** flag if PD ≥ 1 ÷ (1 + {cost_ratio}) = {pct(optimal_threshold)}
""")

    st.divider()
    st.markdown("**The full scorecard**: every account's score is the sum of one row from each feature")
    short = {"limit_bal": "Credit limit", "pay_status_latest": "Repayment status",
             "months_delinquent_6m": "Months late (of 6)", "avg_utilization_6m": "Utilization (6 mo)"}
    sc = b.scorecard
    cols = st.columns(2) + st.columns(2)   # 2 x 2 grid so every table has room
    for c, f in zip(cols, FEATURES):
        part = sc[sc["feature"] == f][["label", "points"]].rename(columns={"label": short[f], "points": "Points"})
        c.dataframe(part.style.format({"Points": "{:.0f}"}), hide_index=True, width="stretch")
    st.caption("Risk bands by PD: " + " · ".join(f"**{bd}** {lo:.0%}–{min(hi, 1):.0%}" for bd, lo, hi, _ in RISK_BANDS))


# ==========================================================================
# TAB 2: CUT-OFF STRATEGY
# ==========================================================================
with tab_cutoff:
    st.subheader("How many accounts should we flag?")
    st.markdown(
        f"All results are on the **{len(test):,} test accounts** the model never saw. "
        "Move the cut-off to see the trade-off between catching defaulters and flagging good customers.")

    threshold = st.slider("Cut-off: flag accounts with PD at or above", 2, 80, int(round(optimal_threshold * 100)),
                          format="%d%%", help="Starts at the cost-optimal cut-off from the sidebar, rounded to a whole %.") / 100

    flag = pd_test >= threshold
    tp = int(((flag) & (y_test == 1)).sum()); fp = int(((flag) & (y_test == 0)).sum())
    fn = int(((~flag) & (y_test == 1)).sum()); tn = int(((~flag) & (y_test == 0)).sum())
    cost = cost_ratio * fn + fp
    cost_nobody = cost_ratio * (tp + fn)
    cost_everyone = tn + fp

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts flagged", pct(flag.mean()), help=f"{flag.sum():,} of {len(flag):,}")
    k2.metric("Defaulters caught", pct(tp / (tp + fn)), help="Recall: share of all defaulters that were flagged.")
    k3.metric("Flagged accounts that default", pct(tp / max(tp + fp, 1)), help="Precision.")
    k4.metric("Cost vs. flagging nobody", pct(cost / cost_nobody - 1, 0), delta_color="off",
              help=f"Total cost {cost:,} ({cost_ratio} × {fn} missed + {fp} false alarms) vs {cost_nobody:,} if no account is flagged.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        # Cost of every possible cut-off
        grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
        costs = [cost_ratio * ((pd_test < g) & (y_test == 1)).sum() + ((pd_test >= g) & (y_test == 0)).sum() for g in grid]
        fig = go.Figure()
        fig.add_scatter(x=grid, y=costs, mode="lines", line=dict(color=NAVY, width=3), name="Total cost")
        fig.add_vline(x=optimal_threshold, line=dict(color=GREEN, dash="dash"),
                      annotation_text=f"Cost-optimal {pct(optimal_threshold)}", annotation_position="top right")
        fig.add_scatter(x=[threshold], y=[cost], mode="markers", marker=dict(size=13, color=AMBER), name="Your cut-off")
        fig.update_layout(title=f"Total cost at each cut-off (missed default = {cost_ratio}× false alarm)",
                          xaxis_title="Cut-off (PD)", yaxis_title="Cost (false-alarm units)", xaxis_tickformat=".0%",
                          height=380, margin=dict(t=50), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("**Outcomes at your cut-off**")
        st.dataframe(pd.DataFrame(
            {"Flagged": [f"{tp:,} defaulters caught", f"{fp:,} false alarms"],
             "Not flagged": [f"{fn:,} defaulters missed", f"{tn:,} correctly left alone"]},
            index=["Defaulted", "Did not default"]), width="stretch")
        st.markdown(f"""
- **Flagging nobody** costs {cost_nobody:,} (every defaulter missed).
- **Flagging everyone** costs {cost_everyone:,} (every good customer contacted).
- **Your cut-off** costs **{cost:,}**: {cost_ratio} × {fn:,} missed + {fp:,} false alarms.
""")
        if abs(threshold - optimal_threshold) > 0.01:
            st.info(f"The cost-optimal cut-off for a {cost_ratio}× cost ratio is {pct(optimal_threshold)}. "
                    "Calibrated PDs make this formula work: see the Model health tab.")

    # Gains chart
    order = np.argsort(-pd_test)
    caught = np.cumsum(y_test[order]) / y_test.sum()
    share = np.arange(1, len(order) + 1) / len(order)
    fig = go.Figure()
    fig.add_scatter(x=share, y=caught, mode="lines", line=dict(color=NAVY, width=3), name="Ranking by PD")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="Random")
    fig.add_scatter(x=[flag.mean()], y=[tp / (tp + fn)], mode="markers", marker=dict(size=13, color=AMBER), name="Your cut-off")
    fig.update_layout(title="Gains: share of defaulters caught vs share of accounts flagged",
                      xaxis_title="Accounts flagged (riskiest first)", yaxis_title="Defaulters caught",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=380, margin=dict(t=50),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")


# ==========================================================================
# TAB 3: MODEL HEALTH
# ==========================================================================
with tab_health:
    st.subheader("Is the model accurate, calibrated, fair and stable?")
    challengers, cv_auc = load_challengers(b)
    auc = roc_auc_score(y_test, pd_test)

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Test AUC", f"{auc:.3f}", help=f"Cross-validated AUC on training data: {cv_auc:.3f}")
    h2.metric("Gini", f"{2 * auc - 1:.3f}", help="2 × AUC − 1, the measure most banks report.")
    h3.metric("KS", f"{ks_statistic(y_test, pd_test):.3f}",
              help="Largest gap between cumulative share of defaulters and of good accounts, riskiest first.")
    h4.metric(f"Average PD (actual {y_test.mean():.1%})", f"{pd_test.mean():.1%}", help="Overall calibration: average predicted PD vs actual default rate.")
    h5.metric("Brier score", f"{brier_score_loss(y_test, pd_test):.3f}", help="Mean squared error of the PD. Lower is better.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fpr, tpr, _ = roc_curve(y_test, pd_test)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=NAVY, width=3), name=f"Scorecard (AUC {auc:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="Random")
        fig.update_layout(title="ROC curve (test set)", xaxis_title="Good accounts flagged",
                          yaxis_title="Defaulters caught", height=380, margin=dict(t=50), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with c2:
        dec = decile_table(y_test, pd_test).reset_index()
        fig = go.Figure()
        fig.add_bar(x=dec["decile"], y=dec["avg_predicted_pd"], name="Average predicted PD", marker_color=NAVY)
        fig.add_bar(x=dec["decile"], y=dec["actual_default_rate"], name="Actual default rate", marker_color=AMBER)
        fig.update_layout(title="Calibration by decile (1 = riskiest 10%)", barmode="group", yaxis_tickformat=".0%",
                          xaxis_title="Decile", height=380, margin=dict(t=50), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    st.caption("The riskiest deciles are well calibrated. In the safer half, predicted PDs sit in a narrow band and "
               "actual rates scatter around it: four features can't separate 'clean' accounts much further.")

    st.divider()
    st.markdown("**Challenger models**: what does the scorecard's transparency cost?")
    st.dataframe(challengers.style.format({"CV AUC": "{:.3f}", "Test AUC": "{:.3f}", "Test Gini": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("Gradient boosting gains about 6 points of Gini but can't be expressed as points or reason codes. "
               "Typical practice: deploy the scorecard as champion and monitor the gap with the challenger.")

    st.divider()
    st.markdown("**Fairness check**: sex and age are never given to the model, so outcomes are checked after the fact")
    f = test.assign(flagged=pd_test >= optimal_threshold)
    f["Group"] = f["SEX"].map({1: "Male", 2: "Female"})
    f["Age band"] = pd.cut(f["AGE"], [0, 29, 39, 49, 120], labels=["Under 30", "30–39", "40–49", "50+"])
    fair = []
    for col in ["Group", "Age band"]:
        g = f.groupby(col, observed=True).agg(Accounts=("pd", "size"), Predicted=("pd", "mean"),
                                              Actual=(TARGET, "mean"), Flagged=("flagged", "mean"))
        g.index = [str(i) for i in g.index]
        fair.append(g)
    fair = pd.concat(fair)
    fair["Predicted − actual"] = fair["Predicted"] - fair["Actual"]
    st.dataframe(fair.style.format({"Predicted": "{:.1%}", "Actual": "{:.1%}", "Flagged": "{:.1%}",
                                    "Predicted − actual": "{:+.1%}"}), width="stretch")
    st.caption(f"Flag rates use the current cut-off ({pct(optimal_threshold)}). Each group's predicted PD is within about "
               "2 points of its actual default rate.")

    st.divider()
    st.markdown("**Drift monitoring (PSI)**: simulate a deteriorating population and see whether monitoring would catch it")
    s1, s2 = st.columns(2)
    util_up = s1.slider("Increase in utilization", 0, 100, 30, step=5, format="+%d%%")
    slip = s2.slider("Share of current customers slipping 1 month late", 0, 50, 10, step=5, format="%d%%")
    rng = np.random.default_rng(7)
    stressed = test[FEATURES].copy()
    stressed["avg_utilization_6m"] *= 1 + util_up / 100
    slipped = (stressed["pay_status_latest"] <= 0) & (rng.random(len(stressed)) < slip / 100)
    stressed.loc[slipped, "pay_status_latest"] = 1
    stressed.loc[slipped, "months_delinquent_6m"] += 1
    stressed_pd = b.predict_pd(stressed)
    table = psi(train_scores, pd_to_score(stressed_pd))
    total_psi = table["psi"].sum()
    status_txt = "Stable" if total_psi < 0.10 else ("Investigate" if total_psi < 0.25 else "Significant shift")
    p1, p2, p3 = st.columns(3)
    p1.metric("PSI", f"{total_psi:.3f}", help="< 0.10 stable, 0.10–0.25 investigate, > 0.25 significant shift.")
    p2.metric("Status", status_txt)
    p3.metric("Average PD", pct(stressed_pd.mean()), delta=f"{(stressed_pd.mean() - pd_test.mean()) * 100:+.1f} pts",
              delta_color="inverse")
    fig = go.Figure()
    x = [f"{i + 1}" for i in range(len(table))]
    fig.add_bar(x=x, y=table["expected_pct"], name="Training (expected)", marker_color=GREY)
    fig.add_bar(x=x, y=table["actual_pct"], name="Stressed (actual)", marker_color=AMBER)
    fig.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", margin=dict(t=30),
                      xaxis_title="Score band (training deciles, 1 = lowest scores)", legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.caption("PSI = Σ (actual% − expected%) × ln(actual% ÷ expected%). Small deteriorations can raise the average PD "
               "while PSI stays under 0.10, which is why PSI is monitored alongside actual default rates.")

    with st.expander("Limitations and how this model would be owned in production"):
        st.markdown("""
**Limitations**
- **Data:** one bank in Taiwan, 2005, during a card-debt crisis; amounts in NT$. The method transfers, the numbers don't.
- **Horizon:** predicts default next month; regulatory PD models (IFRS 9 / Basel) use 12-month and lifetime horizons.
- **Behavioural only:** scores existing customers; it can't approve new applicants with no repayment history.
- **Assumption:** repayment codes −2 and 0 are interpreted as "no use" and "revolving"; this needs confirming with the data owner.

**Model ownership**
| Area | Practice |
|---|---|
| Documentation | Model card: purpose, data, exclusions, assumptions, performance, limitations, intended and prohibited uses |
| Validation | Independent review of binning, WoE, coefficients, calibration and the fairness analysis |
| Monitoring | Monthly score and feature PSI; quarterly AUC/Gini, KS, decile calibration and the fairness table |
| Triggers | PSI > 0.25, Gini more than 5 points below development, or calibration drift beyond tolerance |
| Challenger | Gradient-boosting model kept in parallel to track the accuracy gap |
| Governance | Version control, change log, alignment with model risk policy (e.g. OSFI Guideline E-23) |
""")

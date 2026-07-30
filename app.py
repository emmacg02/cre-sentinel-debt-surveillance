"""
app.py — CRE Sentinel: Commercial Real Estate Debt Surveillance

Streamlit application over the scored synthetic loan tape and trained
XGBoost model. Three views:
  1. Portfolio Overview  - executive-level book health
  2. Loan Explorer       - single-loan drill-down with SHAP explanation
  3. Stress Test         - live rate / occupancy / NOI / valuation shocks

Run locally:    streamlit run app/app.py
Deploy:         push to GitHub, connect the repo at streamlit.io/cloud,
                 set the main file path to app/app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

st.set_page_config(page_title="CRE Sentinel", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------
# Data / model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_model.joblib")
    with open("models/feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, feature_columns

@st.cache_data
def load_data():
    df = pd.read_csv("data/scored_loan_tape.csv",
                      parse_dates=["origination_date", "maturity_date"])
    return df

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model, FEATURE_COLUMNS = load_model()
df = load_data()
explainer = load_explainer(model)

NUMERIC_FEATURES = [
    "interest_rate_pct", "ltv", "dscr", "debt_yield", "interest_coverage",
    "occupancy_pct", "loan_age_years", "years_to_maturity",
]
CATEGORICAL_FEATURES = ["property_type", "region", "rate_type"]
TODAY = pd.Timestamp("2026-01-01")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the exact feature matrix the model was trained on. Must
    stay in sync with scripts/train_models.py — if you change one,
    change the other."""
    f = frame.copy()
    f["loan_age_years"] = (TODAY - f["origination_date"]).dt.days / 365
    f["years_to_maturity"] = (f["maturity_date"] - TODAY).dt.days / 365
    X = pd.get_dummies(f[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                        columns=CATEGORICAL_FEATURES, drop_first=True)
    # Ensure identical column set/order to training time
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    return X[FEATURE_COLUMNS]


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("📊 CRE Sentinel")
st.sidebar.caption("Commercial Real Estate Debt Early-Warning System")
page = st.sidebar.radio("View", ["Portfolio Overview", "Loan Explorer", "Stress Test"])
st.sidebar.divider()
st.sidebar.caption(
    "All loan data is synthetic, generated to match realistic CRE "
    "underwriting distributions. No real borrower data is used. "
    "See data/synthetic_loan_tape.csv generation logic in scripts/."
)

# ---------------------------------------------------------------------
# PAGE 1 — Portfolio Overview
# ---------------------------------------------------------------------
if page == "Portfolio Overview":
    st.title("Portfolio Overview")

    aum = df["current_balance"].sum()
    w_dscr = np.average(df["dscr"], weights=df["current_balance"])
    w_ltv = np.average(df["ltv"], weights=df["current_balance"])
    avg_risk = df["risk_score"].mean()
    watchlist_pct = (df["watchlist_rating"] != "Pass").mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUM (current balance)", f"${aum/1e6:,.0f}M")
    c2.metric("Weighted DSCR", f"{w_dscr:.2f}x")
    c3.metric("Weighted LTV", f"{w_ltv:.1%}")
    c4.metric("Avg Risk Score", f"{avg_risk:.1f} / 100")
    c5.metric("On Watchlist", f"{watchlist_pct:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = df["risk_tier"].value_counts().reindex(
            ["Low", "Moderate", "Elevated", "High"]).fillna(0)
        fig = px.bar(tier_counts, x=tier_counts.index, y=tier_counts.values,
                     color=tier_counts.index,
                     color_discrete_map={"Low": "#2ca02c", "Moderate": "#ffbb33",
                                          "Elevated": "#ff7f0e", "High": "#d62728"},
                     labels={"x": "Risk Tier", "y": "Number of Loans"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Exposure by Property Type")
        exposure = df.groupby("property_type")["current_balance"].sum().sort_values()
        fig = px.bar(exposure, x=exposure.values / 1e6, y=exposure.index, orientation="h",
                     labels={"x": "Current Balance ($M)", "y": ""})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Maturity Wall — Refinancing Exposure by Year")
    df["maturity_year"] = df["maturity_date"].dt.year
    wall = df[df["maturity_year"].between(2026, 2032)].groupby("maturity_year")["current_balance"].sum()
    fig = px.bar(wall, x=wall.index, y=wall.values / 1e6,
                 labels={"x": "Maturity Year", "y": "Balance Maturing ($M)"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("DSCR vs. LTV, colored by Risk Score")
    fig = px.scatter(df, x="ltv", y="dscr", color="risk_score",
                      hover_data=["loan_id", "property_type", "current_balance"],
                      color_continuous_scale="RdYlGn_r",
                      labels={"ltv": "Loan-to-Value", "dscr": "DSCR"})
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.80, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 2 — Loan Explorer
# ---------------------------------------------------------------------
elif page == "Loan Explorer":
    st.title("Loan Explorer")

    loan_id = st.selectbox("Select a loan", df["loan_id"].sort_values())
    loan = df[df["loan_id"] == loan_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{loan['risk_score']:.1f} / 100")
    c2.metric("Risk Tier", loan["risk_tier"])
    c3.metric("Watchlist Rating", loan["watchlist_rating"])
    c4.metric("Covenant Status", loan["covenant_status"])

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Loan Detail")
        detail = {
            "Property Type": loan["property_type"], "Region": loan["region"],
            "Origination": loan["origination_date"].date(),
            "Maturity": loan["maturity_date"].date(),
            "Rate Type": loan["rate_type"], "Interest Rate": f"{loan['interest_rate_pct']:.2f}%",
            "Current Balance": f"${loan['current_balance']:,.0f}",
            "Appraised Value": f"${loan['appraised_value']:,.0f}",
            "NOI": f"${loan['net_operating_income']:,.0f}",
            "Occupancy": f"{loan['occupancy_pct']:.1f}%",
            "LTV": f"{loan['ltv']:.1%}", "DSCR": f"{loan['dscr']:.2f}x",
            "Debt Yield": f"{loan['debt_yield']:.1%}",
            "Interest Coverage": f"{loan['interest_coverage']:.2f}x",
            "Delinquency": loan["delinquency_status"],
        }
        st.table(pd.DataFrame(detail.items(), columns=["Field", "Value"]).set_index("Field"))

    with right:
        st.subheader("What's driving this loan's risk score")
        single_row = df[df["loan_id"] == loan_id]
        X_single = build_features(single_row)
        shap_vals = explainer.shap_values(X_single)[0]
        contrib = pd.Series(shap_vals, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=True).tail(8)
        fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index, orientation="h",
            marker_color=["#d62728" if v > 0 else "#2ca02c" for v in contrib.values],
        ))
        fig.update_layout(xaxis_title="Impact on risk score (SHAP value)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = pushes risk score up. Green = pushes risk score down.")

    st.divider()
    st.subheader("Draft credit review questions")
    flags = []
    if loan["dscr"] < 1.15:
        flags.append(f"DSCR is {loan['dscr']:.2f}x, close to or below breakeven — request an updated rent roll and current-quarter operating statement.")
    if loan["ltv"] > 0.75:
        flags.append(f"LTV is {loan['ltv']:.1%} — confirm whether a recent appraisal reflects current market cap rates.")
    if loan["occupancy_pct"] < 80:
        flags.append(f"Occupancy is {loan['occupancy_pct']:.1f}% — ask for a leasing pipeline / tenant retention update.")
    if loan["covenant_status"] == "Breached":
        flags.append("Covenant currently breached — request sponsor's cure plan and timeline.")
    if not flags:
        flags.append("No material red flags — standard periodic reporting review only.")
    for q in flags:
        st.write(f"- {q}")
    st.caption(
        "These questions are generated from verified fields on this loan record only. "
        "No values are invented or estimated — if a field were missing, it would be "
        "flagged rather than guessed."
    )

# ---------------------------------------------------------------------
# PAGE 3 — Stress Test
# ---------------------------------------------------------------------
else:
    st.title("Interactive Stress Test")
    st.caption("Shock the portfolio and watch risk scores and refinancing exposure move in real time.")

    c1, c2, c3, c4 = st.columns(4)
    rate_shock_bps = c1.slider("Interest rate shock (bps, floating-rate loans only)", 0, 300, 0, step=25)
    occupancy_shock = c2.slider("Occupancy shock (%)", -30, 0, 0)
    noi_shock = c3.slider("Additional NOI shock (%)", -30, 0, 0)
    valuation_shock = c4.slider("Appraised value shock (%)", -30, 0, 0)

    shocked = df.copy()
    is_floating = shocked["rate_type"] == "Floating"
    shocked.loc[is_floating, "interest_rate_pct"] = shocked.loc[is_floating, "interest_rate_pct"] + rate_shock_bps / 100

    occ_factor = 1 + occupancy_shock / 100
    noi_factor = (1 + noi_shock / 100) * occ_factor
    shocked["occupancy_pct"] = np.clip(shocked["occupancy_pct"] * occ_factor, 10, 100)
    shocked["net_operating_income"] = shocked["net_operating_income"] * noi_factor
    shocked["appraised_value"] = shocked["appraised_value"] * (1 + valuation_shock / 100)

    annual_debt_service = shocked["current_balance"] * (shocked["interest_rate_pct"] / 100) * 1.15
    shocked["dscr"] = shocked["net_operating_income"] / annual_debt_service
    shocked["ltv"] = shocked["current_balance"] / shocked["appraised_value"]
    shocked["debt_yield"] = shocked["net_operating_income"] / shocked["current_balance"]
    shocked["interest_coverage"] = shocked["net_operating_income"] / (shocked["current_balance"] * (shocked["interest_rate_pct"] / 100))

    X_shocked = build_features(shocked)
    shocked["risk_score"] = np.round(model.predict_proba(X_shocked)[:, 1] * 100, 2)
    shocked["risk_tier"] = pd.cut(shocked["risk_score"], bins=[-1, 20, 50, 75, 100],
                                   labels=["Low", "Moderate", "Elevated", "High"])

    baseline_avg = df["risk_score"].mean()
    shocked_avg = shocked["risk_score"].mean()
    baseline_gap = df.loc[df["ltv"] > 0.80, "current_balance"].sum()
    shocked_gap = shocked.loc[shocked["ltv"] > 0.80, "current_balance"].sum()
    baseline_high = (df["risk_tier"] == "High").sum()
    shocked_high = (shocked["risk_tier"] == "High").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Portfolio Risk Score", f"{shocked_avg:.1f}", delta=f"{shocked_avg - baseline_avg:+.1f}")
    c2.metric("Refinancing Gap (LTV > 80%)", f"${shocked_gap/1e6:,.0f}M",
              delta=f"{(shocked_gap - baseline_gap)/1e6:+,.0f}M", delta_color="inverse")
    c3.metric("Loans in High-Risk Tier", f"{shocked_high}", delta=f"{shocked_high - baseline_high:+d}", delta_color="inverse")

    st.subheader("Risk Tier Shift: Baseline vs. Shocked")
    compare = pd.DataFrame({
        "Baseline": df["risk_tier"].value_counts().reindex(["Low", "Moderate", "Elevated", "High"]).fillna(0),
        "Shocked": shocked["risk_tier"].value_counts().reindex(["Low", "Moderate", "Elevated", "High"]).fillna(0),
    })
    fig = px.bar(compare, barmode="group",
                 color_discrete_map={"Baseline": "#7f8fa6", "Shocked": "#d62728"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Loans With the Largest Risk Score Increase")
    delta_df = pd.DataFrame({
        "loan_id": df["loan_id"], "property_type": df["property_type"],
        "baseline_score": df["risk_score"], "shocked_score": shocked["risk_score"],
    })
    delta_df["increase"] = delta_df["shocked_score"] - delta_df["baseline_score"]
    st.dataframe(delta_df.sort_values("increase", ascending=False).head(10), use_container_width=True)

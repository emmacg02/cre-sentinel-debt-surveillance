# CRE Sentinel — Commercial Real Estate Debt Early-Warning System

A portfolio-surveillance and credit-risk analytics project for commercial real estate (CRE) lending. Built to simulate how a lender monitors a loan book across office, multifamily, retail, industrial, hospitality, and mixed-use properties, and flags which assets need attention before they default or hit a maturity wall.

## Business problem

CRE lenders manage hundreds of loans at once. Loan officers and credit committees need to know, at any point in time:

- Which loans are becoming vulnerable
- Which borrowers require immediate review
- Where refinancing risk is concentrated
- How rate, occupancy, and NOI shocks would move the portfolio
- Which assets belong on a watchlist

Most lenders answer this with static, point-in-time reviews of borrower financials. This project builds a continuous surveillance layer instead: a scored, monitored, and stress-tested view of the whole book, updated as conditions change.

## What this project builds

- A synthetic commercial real estate loan database (~500–1,000 loans)
- A credit-risk model that scores each loan 0–100
- A Power BI portfolio dashboard
- An interactive stress-testing application (Streamlit)
- An AI-assisted, fully verified credit-risk memo generator

## Data

### Real data
Macro series pulled programmatically from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/) (JSON): interest rates (SOFR), CPI, unemployment, and commercial property price indices.

### Synthetic data
The loan tape itself (~500–1,000 individual loans) is synthetic. Real CRE loan tapes contain borrower-identifying, NDA-protected financial data that cannot be published or used in a public portfolio project. The synthetic tape is generated to match realistic distributions — default rates, capital structures, DSCR/LTV ranges by property type — but contains no real borrowers, properties, or lenders.

This is documented explicitly in `data_dictionary.md`, including:
- How each field was generated
- What assumptions were used
- Which fields are real (macro) vs. simulated (loan-level)
- Why simulation was necessary

**Core loan-level fields:** LTV, DSCR, debt yield, interest coverage, occupancy, NOI, maturity date, interest-rate structure, covenant status, refinancing gap, delinquency status, watchlist rating, probability of default.

## Modeling

Two models, trained and compared honestly rather than picking a winner on accuracy alone:

- **Logistic regression** — the explainable baseline
- **XGBoost** — captures non-linear interactions (e.g., an office asset hit by both a rate spike and an occupancy drop)

```python
features = ['DSCR', 'LTV', 'Debt_Yield', 'Interest_Coverage', 'Occupancy_Pct']
X = loan_tape_df[features]

model = xgb.XGBClassifier().fit(X_train, y_train)
loan_tape_df['risk_score'] = np.round(model.predict_proba(X)[:, 1] * 100, 2)
```

### What the model card actually documents

The point of this section isn't a leaderboard accuracy number. It's showing I understand where credit models break:

- **Class imbalance** — CRE default rates run ~2–5%. Addressed with SMOTE / `scale_pos_weight`, and reported with precision/recall/F1 rather than accuracy alone.
- **Data leakage** — post-default fields (legal/collections timestamps) are excluded from training features.
- **False positives** — flags a technically-breached but healthy sponsor (e.g., late reporting), risking loss of a good customer.
- **False negatives** — misses a borrower who injects short-term capital to mask a failing DSCR right before default.
- **Model limitations** — the model can't see sponsor reputation, pending lease negotiations, or local zoning changes. It supports underwriting; it doesn't replace it.

### Stress tests

- **Rate shocks:** +50bps / +100bps / +200bps on floating-rate loans → recomputed interest coverage
- **Operating shocks:** NOI and occupancy down 10–30% → recomputed DSCR
- **Valuation shocks:** appraisal haircuts → recomputed LTV and refinancing gap at maturity

## Power BI dashboard

Seven pages:

1. Executive portfolio overview (AUM, weighted DSCR/LTV, delinquency)
2. Maturity wall (12–36 month refinancing exposure)
3. Geographic and property-type concentration
4. Watchlist and covenant monitoring
5. Interactive stress testing (rate / NOI / occupancy sliders)
6. Individual loan review
7. Model explanation (SHAP)

## Credit Review Copilot

An LLM-assisted memo drafter, scoped deliberately narrowly:

- Takes a selected loan ID
- Pulls only verified values from the database (no web search, no free-form recall)
- Summarizes principal risk drivers and what changed since the last period
- Drafts borrower follow-up questions
- Produces a preliminary credit memo for human review

**Constraints, by design:**
- If a field is missing, it reports the gap — it does not estimate or invent a number.
- It has no write access to loan status, covenants, or approvals. It drafts; it does not decide.

## Deliverables

- Live Streamlit app
- Power BI dashboard (`.pbix`)
- This repository (documented pipelines, SQL init scripts, requirements)
- `data_dictionary.md`
- `model_card.md`
- 6–8 page credit-risk report (PDF)
- 2-minute demo video
- AI prompt/evaluation log
- LinkedIn carousel summarizing the problem and solution

## Stack

Python (pandas, scikit-learn, XGBoost, SHAP) · PostgreSQL · Streamlit · Power BI · FRED API

## Disclaimer

All loan-level data in this repository is synthetic and generated for demonstration purposes. No real borrower, property, or lender information is used or represented.

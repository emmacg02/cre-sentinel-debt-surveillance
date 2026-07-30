# CRE Sentinel — Commercial Real Estate Debt Early-Warning System

A portfolio-surveillance and credit-risk analytics project for commercial real estate (CRE) lending. Built to simulate how a lender monitors a loan book across office, multifamily, retail, industrial, hospitality, and mixed-use properties, and flags which assets need attention before they default or hit a maturity wall.

**🔗 Live app:** [cre-sentinel-debt-surveillance.streamlit.app](https://cre-sentinel-debt-surveillance.streamlit.app)

## Business problem

CRE lenders manage hundreds of loans at once. Loan officers and credit committees need to know, at any point in time:

- Which loans are becoming vulnerable
- Which borrowers require immediate review
- Where refinancing risk is concentrated
- How rate, occupancy, and NOI shocks would move the portfolio
- Which assets belong on a watchlist

Most lenders answer this with static, point-in-time reviews of borrower financials. This project builds a continuous surveillance layer instead: a scored, monitored, and stress-tested view of the whole book, updated as conditions change.

## What this project builds

- A synthetic commercial real estate loan database (1,000 loans)
- A credit-risk model that scores each loan 0–100
- A Power BI portfolio dashboard
- An interactive stress-testing application (Streamlit) — **[try it live](https://cre-sentinel-debt-surveillance.streamlit.app)**
- An AI-assisted, fully verified credit-risk memo generator

## Data

### Real data
Macro series pulled programmatically from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/) (JSON): interest rates (SOFR), CPI, unemployment, and commercial property price indices.

### Synthetic data
The loan tape itself (1,000 individual loans) is synthetic. Real CRE loan tapes contain borrower-identifying, NDA-protected financial data that cannot be published or used in a public portfolio project. The synthetic tape is generated to match realistic distributions — default rates, capital structures, DSCR/LTV ranges by property type — but contains no real borrowers, properties, or lenders.

Full provenance, generation logic, and field-by-field origin: **[data_dictionary.md](reports/data_dictionary.md)**

**Core loan-level fields:** LTV, DSCR, debt yield, interest coverage, occupancy, NOI, maturity date, interest-rate structure, covenant status, refinancing gap, delinquency status, watchlist rating, probability of default.

## Modeling

Two models, trained and compared honestly rather than picking a winner on accuracy alone:

- **Logistic regression** — the explainable baseline
- **XGBoost** — captures non-linear interactions (e.g., an office asset hit by both a rate spike and an occupancy drop)

On this dataset, logistic regression actually beats XGBoost on PR-AUC and recall — a known small-sample effect. Full honest comparison and the reasoning for keeping XGBoost as the production model anyway: **[model_card.md](reports/model_card.md)**

### Stress tests

Live in the app: rate shocks (floating-rate loans), occupancy shocks, NOI shocks, and appraised value shocks — the whole portfolio re-scores in real time.

## Power BI dashboard

Seven pages: Executive Overview, Maturity Wall, Geographic/Property Exposure, Watchlist & Covenant Monitoring, Stress Testing, Individual Loan Review, Model Explanation.
*(dashboard file and screenshots to be added — see `reports/` once published)*

## Credit Review Copilot

An LLM-assisted memo drafter, scoped deliberately narrowly — grounded only in verified database fields, with no authority to approve, decline, or reprice a loan. Full prompt design and evaluation runs, including a caught-and-fixed hallucination case: **[ai_prompt_evaluation_log.md](reports/ai_prompt_evaluation_log.md)**

## Deliverables

| Deliverable | Link |
|---|---|
| 🟢 Live Streamlit app | [cre-sentinel-debt-surveillance.streamlit.app](https://cre-sentinel-debt-surveillance.streamlit.app) |
| 📄 Credit-Risk Report (PDF) | [reports/credit_risk_report.pdf](reports/credit_risk_report.pdf) |
| 📋 Model Card | [reports/model_card.md](reports/model_card.md) |
| 📖 Data Dictionary | [reports/data_dictionary.md](reports/data_dictionary.md) |
| ⚖️ Model Comparison (honest metrics) | [reports/model_comparison.md](reports/model_comparison.md) |
| 🤖 AI Prompt & Evaluation Log | [reports/ai_prompt_evaluation_log.md](reports/ai_prompt_evaluation_log.md) |
| 📊 SHAP Explainability Summary | [reports/shap_summary.png](reports/shap_summary.png) |
| 🎠 LinkedIn Carousel | [reports/linkedin_carousel.pdf](reports/linkedin_carousel.pdf) |
| 🎥 Demo Video | *(link coming soon)* |
| 📈 Power BI Dashboard | *(coming soon)* |
| 🐍 Data generator script | [scripts/generate_loan_tape.py](scripts/generate_loan_tape.py) |
| 🧠 Model training script | [scripts/train_models.py](scripts/train_models.py) |
| 💻 Streamlit app source | [app/app.py](app/app.py) |

## Stack

Python (pandas, scikit-learn, XGBoost, SHAP) · PostgreSQL · Streamlit · Power BI · FRED API

## Disclaimer

All loan-level data in this repository is synthetic and generated for demonstration purposes. No real borrower, property, or lender information is used or represented.

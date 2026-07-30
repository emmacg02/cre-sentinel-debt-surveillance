# Data Dictionary — CRE Sentinel Loan Tape

## Origin of the data

**None of this data is real.** There are no real borrowers, properties, or lenders
represented anywhere in this repository.

Real commercial real estate (CRE) loan tapes contain borrower-identifying financial
information that lenders hold under strict non-disclosure agreements. Publishing or
training a public model on real loan-level data would not be possible without
violating those agreements. This project instead generates a **synthetic loan tape**
(`data/synthetic_loan_tape.csv`) using `scripts/generate_loan_tape.py`, built to
reproduce the statistical relationships of a real CRE book — realistic ratio
distributions by property type, and a default outcome that depends on those ratios
the way an actual credit event would — without any real underlying loan.

The one genuinely real component is the macroeconomic context this project is
designed to be enriched with: interest rate, inflation, and employment series
pulled programmatically from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/)
(a public, government-run economic data source — no NDA or privacy issue applies).

## How the synthetic tape was generated (summary)

1. Each loan is assigned a property type and region.
2. Loan size, interest rate, and occupancy are drawn from distributions calibrated
   per property type (e.g., hospitality carries lower average occupancy and higher
   loan size variance than industrial).
3. NOI, appraised value, DSCR, LTV, debt yield, and interest coverage are then
   **calculated** from those inputs — they are not independently randomized, which
   is what keeps the ratios internally consistent (a loan can't have simultaneously
   inflated income and a discounted appraisal, because the appraisal is derived
   from the same income figure via a cap rate).
4. Default outcome is drawn from a logistic function of DSCR, LTV, occupancy, and a
   property-type risk adjustment, plus random noise — so weak-ratio loans are more
   likely, but not certain, to default. This is what gives a model trained on the
   data something real to learn.

Full generation logic, including every distribution parameter and the exact default
model, is in `scripts/generate_loan_tape.py` — the code is the authoritative source;
this document summarizes it.

## Field reference

### `data/synthetic_loan_tape.csv` (raw, 1,000 loans)

| Field | Type | Origin | Description |
|---|---|---|---|
| `loan_id` | string | Synthetic | Unique loan identifier, `L0001`–`L1000` |
| `property_type` | categorical | Synthetic | Office, Multifamily, Retail, Industrial, Hospitality, Mixed-Use |
| `region` | categorical | Synthetic | Northeast, Southeast, Midwest, Southwest, West |
| `origination_date` | date | Synthetic | 1–7 years before Jan 2026 |
| `maturity_date` | date | Synthetic | Origination + a 3/5/7/10-year term |
| `rate_type` | categorical | Synthetic | Fixed or Floating (60/40 split) |
| `interest_rate_pct` | float | Synthetic (calculated) | Origination interest rate |
| `original_loan_amount` | float ($) | Synthetic | Loan size at origination, log-normal by property type |
| `current_balance` | float ($) | Synthetic (calculated) | Amortized balance as of Jan 2026 |
| `appraised_value` | float ($) | Synthetic (calculated) | `net_operating_income / cap_rate` |
| `net_operating_income` | float ($) | Synthetic (calculated) | Annual property income before debt service |
| `occupancy_pct` | float | Synthetic | Percent of the property leased/occupied |
| `ltv` | float | **Calculated** | `current_balance / appraised_value` |
| `dscr` | float | **Calculated** | `net_operating_income / annual_debt_service` |
| `debt_yield` | float | **Calculated** | `net_operating_income / current_balance` |
| `interest_coverage` | float | **Calculated** | `net_operating_income / (current_balance × interest rate)` |
| `covenant_status` | categorical | Synthetic | Compliant / Breached — probability scaled to default risk |
| `delinquency_status` | categorical | Synthetic | Current / 30-60 Days / 90+ Days |
| `watchlist_rating` | categorical | **Rule-based**, not ML | Pass / Watch / High Risk, from fixed DSCR/LTV/occupancy thresholds — this is what a human credit officer's simple heuristic would flag, kept separate from the ML risk score so the two can be compared |
| `defaulted` | binary | Synthetic target | 1 if the loan defaulted, drawn from the logistic default model described above |

### `data/scored_loan_tape.csv` (adds model output)

Same fields as above, plus:

| Field | Type | Origin | Description |
|---|---|---|---|
| `risk_score` | float, 0–100 | **Model output** | XGBoost predicted default probability × 100 |
| `risk_tier` | categorical | **Derived from risk_score** | Low (0–20) / Moderate (20–50) / Elevated (50–75) / High (75–100) |

## Fields deliberately excluded from the model

`covenant_status`, `delinquency_status`, and `watchlist_rating` are present in the
data for dashboard and reporting purposes but are **not used as model input
features**. They are generated from the same underlying risk signal as the
`defaulted` target itself, so including them in training would let the model read
a proxy of the answer rather than learn from the financial fundamentals — textbook
data leakage. See `model_card.md` for the full discussion.

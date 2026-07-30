"""
generate_loan_tape.py

Creates a synthetic commercial real estate (CRE) loan portfolio for the
CRE Sentinel project. Every loan in this file is fabricated -- there is
no real borrower or lender data here. The point is to build a dataset
that BEHAVES like a real CRE loan book so a risk model can learn
something meaningful from it, without touching any real, confidential
loan data (which lenders keep under strict NDAs).

How this works, in plain terms:
1. We pick a property type and region for each loan (Office, Multifamily,
   Retail, Industrial, Hospitality, Mixed-Use).
2. We give each loan realistic starting numbers: loan size, interest
   rate, appraised property value, how much income the property makes
   (NOI), and how full the building is (occupancy).
3. From those numbers we CALCULATE the financial ratios a real lender
   would look at: LTV, DSCR, debt yield, interest coverage.
4. We then decide whether each loan eventually defaults -- but not
   randomly. A loan with weak ratios (low DSCR, high LTV, low occupancy)
   gets a HIGHER chance of default, and a loan with strong ratios gets a
   LOWER chance. This is what makes the dataset realistic: the model
   will later be able to learn "weak ratios -> more likely to default,"
   which is the whole point of a credit risk model.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Setting a fixed seed means the "random" numbers below always come out
# the same way every time this script runs. That makes the dataset
# reproducible -- anyone who runs this script gets the exact same tape.
np.random.seed(42)

N_LOANS = 1000

PROPERTY_TYPES = ["Office", "Multifamily", "Retail", "Industrial", "Hospitality", "Mixed-Use"]
# Rough relative weights: multifamily and industrial are the largest CRE
# lending segments today, office and hospitality are smaller and riskier.
PROPERTY_WEIGHTS = [0.20, 0.28, 0.16, 0.20, 0.08, 0.08]

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

# Each property type carries a different baseline riskiness in the real
# market right now (post-2023 office stress, resilient industrial/multifamily).
# This number nudges the base default probability up or down by property type.
PROPERTY_RISK_ADJUSTMENT = {
    "Office": 0.12,
    "Hospitality": 0.06,
    "Retail": 0.03,
    "Mixed-Use": 0.02,
    "Multifamily": -0.03,
    "Industrial": -0.04,
}

def random_dates(n):
    """Give each loan an origination date in the last 1-7 years and a
    3-10 year term, so maturity dates are spread across the next several
    years (this is what powers the 'Maturity Wall' view later)."""
    today = datetime(2026, 1, 1)
    origin_days_ago = np.random.randint(365, 365 * 7, size=n)
    origination = [today - timedelta(days=int(d)) for d in origin_days_ago]
    term_years = np.random.choice([3, 5, 7, 10], size=n, p=[0.25, 0.35, 0.25, 0.15])
    maturity = [o + timedelta(days=int(t * 365)) for o, t in zip(origination, term_years)]
    return origination, maturity


def generate_loan_tape(n=N_LOANS):
    property_type = np.random.choice(PROPERTY_TYPES, size=n, p=PROPERTY_WEIGHTS)
    region = np.random.choice(REGIONS, size=n)
    rate_type = np.random.choice(["Fixed", "Floating"], size=n, p=[0.6, 0.4])

    origination, maturity = random_dates(n)

    # Loan size varies a lot by property type (a hotel loan looks very
    # different in scale from a small strip-mall retail loan).
    size_base = {
        "Office": 18_000_000, "Multifamily": 12_000_000, "Retail": 8_000_000,
        "Industrial": 10_000_000, "Hospitality": 22_000_000, "Mixed-Use": 14_000_000,
    }
    original_loan_amount = np.array([
        max(1_000_000, np.random.lognormal(mean=np.log(size_base[pt]), sigma=0.55))
        for pt in property_type
    ])

    # Interest rate: fixed loans priced slightly lower than floating on
    # average at origination; floating loans move with the macro rate
    # environment (that linkage is why we later pull SOFR from FRED).
    base_rate = np.random.normal(6.2, 0.9, size=n)
    interest_rate = np.clip(
        base_rate + np.where(rate_type == "Floating", 0.4, 0.0), 3.5, 11.0
    )

    # Occupancy: how full the building is. Office skews lower right now;
    # multifamily and industrial skew higher.
    occ_base = {
        "Office": 78, "Multifamily": 92, "Retail": 87,
        "Industrial": 94, "Hospitality": 68, "Mixed-Use": 85,
    }
    occupancy_pct = np.clip(
        np.array([np.random.normal(occ_base[pt], 8) for pt in property_type]), 30, 100
    )

    # NOI (Net Operating Income): the property's actual profit before debt
    # payments. Scaled off loan size and occupancy -- a half-empty building
    # makes a lot less money.
    noi_yield = np.random.normal(0.105, 0.015, size=n)  # NOI as % of loan amount, roughly
    net_operating_income = original_loan_amount * noi_yield * (occupancy_pct / 100)

    # Appraised value: derived from NOI using a market cap rate (this is
    # how commercial appraisers actually estimate value).
    cap_rate = np.random.normal(0.065, 0.009, size=n)
    appraised_value = net_operating_income / cap_rate

    # Current balance: assume some loans have amortized a bit since
    # origination.
    years_seasoned = np.array([(datetime(2026, 1, 1) - o).days / 365 for o in origination])
    amortization_factor = np.clip(1 - years_seasoned * 0.012, 0.7, 1.0)
    current_balance = original_loan_amount * amortization_factor

    # --- Calculated ratios (these are the real underwriting metrics) ---
    ltv = current_balance / appraised_value
    annual_debt_service = current_balance * (interest_rate / 100) * 1.15  # rough P&I estimate
    dscr = net_operating_income / annual_debt_service
    debt_yield = net_operating_income / current_balance
    interest_coverage = net_operating_income / (current_balance * (interest_rate / 100))

    # --- Default probability model ---
    # This is the key step: we turn weak ratios into a HIGHER chance of
    # default using a logistic (sigmoid) function, then flip a weighted
    # coin for each loan. This is exactly the kind of relationship a
    # credit model is later trained to rediscover.
    risk_adj = np.array([PROPERTY_RISK_ADJUSTMENT[pt] for pt in property_type])
    logit = (
        -4.4
        + (1.4 - dscr) * 2.8          # DSCR below ~1.4 raises risk sharply
        + (ltv - 0.65) * 3.0          # LTV above ~65% raises risk
        + (85 - occupancy_pct) * 0.02  # low occupancy raises risk
        + risk_adj * 4
        + np.random.normal(0, 0.4, size=n)  # noise: real defaults aren't perfectly predictable
    )
    default_probability = 1 / (1 + np.exp(-logit))
    defaulted = np.random.binomial(1, np.clip(default_probability, 0.005, 0.9))

    # Covenant status and delinquency flow logically from the same
    # underlying weakness -- distressed loans are more likely to have
    # breached a covenant or be delinquent, but it isn't a 1:1 mapping
    # (a loan can breach a covenant without missing a payment yet).
    covenant_breach_prob = np.clip(default_probability * 1.3, 0, 0.95)
    covenant_status = np.where(
        np.random.binomial(1, covenant_breach_prob), "Breached", "Compliant"
    )
    delinquency_status = np.select(
        [defaulted == 1, np.random.binomial(1, default_probability * 0.6) == 1],
        ["90+ Days", "30-60 Days"],
        default="Current",
    )

    # Watchlist rating: a simple, transparent rule-based tier (separate
    # from the ML model) -- this is what a human credit officer would
    # flag by eye, and is useful later to compare against model output.
    watchlist_rating = np.select(
        [
            (dscr < 1.00) | (ltv > 0.85),
            (dscr < 1.15) | (ltv > 0.75) | (occupancy_pct < 72),
        ],
        ["High Risk", "Watch"],
        default="Pass",
    )

    df = pd.DataFrame({
        "loan_id": [f"L{str(i+1).zfill(4)}" for i in range(n)],
        "property_type": property_type,
        "region": region,
        "origination_date": [d.date() for d in origination],
        "maturity_date": [d.date() for d in maturity],
        "rate_type": rate_type,
        "interest_rate_pct": np.round(interest_rate, 2),
        "original_loan_amount": np.round(original_loan_amount, 0),
        "current_balance": np.round(current_balance, 0),
        "appraised_value": np.round(appraised_value, 0),
        "net_operating_income": np.round(net_operating_income, 0),
        "occupancy_pct": np.round(occupancy_pct, 1),
        "ltv": np.round(ltv, 4),
        "dscr": np.round(dscr, 2),
        "debt_yield": np.round(debt_yield, 4),
        "interest_coverage": np.round(interest_coverage, 2),
        "covenant_status": covenant_status,
        "delinquency_status": delinquency_status,
        "watchlist_rating": watchlist_rating,
        "defaulted": defaulted,  # this is the target variable for modeling
    })

    return df


if __name__ == "__main__":
    tape = generate_loan_tape()
    out_path = "/home/claude/cre-sentinel/data/synthetic_loan_tape.csv"
    tape.to_csv(out_path, index=False)
    print(f"Generated {len(tape)} loans -> {out_path}")
    print(f"Default rate: {tape['defaulted'].mean():.2%}")
    print(tape.groupby('property_type')['defaulted'].mean().round(3))

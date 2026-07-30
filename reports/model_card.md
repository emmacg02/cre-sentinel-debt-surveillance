# Model Card — CRE Sentinel Default Risk Model

## Summary

Two models were trained to predict commercial real estate loan default: a
logistic regression baseline and an XGBoost classifier. Both are shipped in this
repository; XGBoost is used as the production scoring model in the app and
dashboard, for reasons explained below — not because it has the better raw
metrics on this dataset, which it does not.

## Training data

- **Source:** synthetic loan tape, 1,000 loans (see `data_dictionary.md` and
  `scripts/generate_loan_tape.py` for full generation logic). No real loan data.
- **Split:** 75% train (750 loans) / 25% test (250 loans), stratified on the
  default label so the imbalance ratio is preserved in both splits.
- **Base rate:** 5.6% of loans defaulted (56 defaults in the full tape, 42 in
  training, 14 in test).

## Features used

`interest_rate_pct`, `ltv`, `dscr`, `debt_yield`, `interest_coverage`,
`occupancy_pct`, `loan_age_years`, `years_to_maturity`, plus one-hot encoded
`property_type`, `region`, and `rate_type`. 18 features total after encoding.

## Features deliberately excluded (data leakage)

`covenant_status`, `delinquency_status`, and `watchlist_rating` were excluded
from the feature set. In the synthetic data generator, these fields are drawn
from the *same* underlying default-risk signal used to generate the `defaulted`
label itself — including them as model inputs would let the model partially
read the answer rather than learn from the underlying financial ratios. In a
real deployment, `delinquency_status` in particular would frequently only be
known concurrently with or after the event you are trying to predict ahead of
time, which is the textbook definition of leakage: a feature that would not
actually be available at the point you need the prediction.

## Class imbalance handling

The default rate is ~5.6%, so the trivial "always predict no default"
classifier would already be 94% "accurate" — which is why accuracy is not
reported anywhere in this project. Two different, legitimate re-weighting
approaches were used instead of accuracy:

- **Logistic regression:** `class_weight="balanced"`, which reweights the loss
  function inversely proportional to class frequency. No synthetic data points
  are created.
- **XGBoost:** `scale_pos_weight = (negative count / positive count)`, the
  equivalent reweighting mechanism built into gradient boosting.

**SMOTE (synthetic minority oversampling) was deliberately not used.** On a
dataset this size, with only ~42 real positive examples in the training fold,
SMOTE-generated synthetic minority points can sit unrealistically close to real
ones and inflate validation performance without adding genuine signal. Class
weighting is the more defensible default here; SMOTE would be worth
revisiting on a larger real loan book with a richer feature set.

## Evaluation results (held-out test set, 250 loans)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.788 | 0.739 |
| PR-AUC (average precision) | **0.328** | 0.285 |
| F1 @ 0.5 threshold | 0.231 | 0.267 |
| Precision @ 0.5 | 0.141 | 0.250 |
| Recall @ 0.5 | **0.643** | 0.286 |

PR-AUC, not ROC-AUC, is the more informative metric here given the ~5.6% class
imbalance — ROC-AUC can look deceptively strong on imbalanced data because it's
dominated by the majority class. On PR-AUC, **logistic regression outperforms
XGBoost on this dataset.**

### Why report a result where the "simpler" model wins

Because it's true, and because papering over it would be a worse signal to send
than the result itself. This is a well-documented small-sample effect: gradient
boosting needs enough positive examples to find non-linear interaction patterns
a linear model can't; with only 42 defaults in training, XGBoost doesn't have
that yet. Full discussion and the resulting production model justification is
in `reports/model_comparison.md`. XGBoost was still shipped as the app's
production model, but the choice is defended on explainability grounds (its
SHAP integration powers the Loan Explorer page) and on expected behavior as the
loan tape scales up — not on this dataset's raw metrics.

## Confusion matrix at the 0.5 threshold (test set)

**Logistic Regression** — predicted vs. actual:
```
                 Predicted: No Default   Predicted: Default
Actual: No Default        181                    55
Actual: Default              5                     9
```

**XGBoost:**
```
                 Predicted: No Default   Predicted: Default
Actual: No Default        224                    12
Actual: Default             10                     4
```

Logistic regression catches more true defaults (9 of 14) but at the cost of far
more false alarms (55 healthy loans flagged). XGBoost is more conservative:
fewer false alarms (12), but it also misses more real defaults (10 of 14
undetected). Neither is free — see the false positive/false negative
discussion below.

## Explainability

XGBoost predictions are explained using SHAP (SHapley Additive exPlanations)
`TreeExplainer`. The global summary (`reports/shap_summary.png`) and per-loan
breakdowns (in the Streamlit app's Loan Explorer page) show that **LTV,
occupancy, and debt yield are the dominant drivers** of predicted risk, with
interest rate and loan age as secondary factors — directionally consistent
with how the synthetic default labels were generated, which is a useful sanity
check that the model learned the intended relationships rather than something
spurious.

## Why this model should not make automated credit decisions

This is the most important section of this document, not a disclaimer tacked
on at the end.

- **False positives (flagging a healthy loan):** a temporarily depressed DSCR
  from a delayed reporting deadline, a highly capitalized sponsor's brief
  technical covenant breach, or a seasonal occupancy dip in a resilient asset
  can all trigger a high risk score. Acting on the score alone risks
  needlessly damaging a profitable, low-risk relationship.
- **False negatives (missing a real default):** a distressed borrower
  temporarily injecting outside capital to mask a failing DSCR right before a
  structural default is invisible to a model that only sees the reported
  ratios. This project's confusion matrices show this is a real, measured
  risk, not a theoretical one — both models missed real defaults in testing.
- **What the model cannot see at all:** sponsor reputation and track record,
  pending lease negotiations not yet reflected in occupancy figures, local
  zoning or regulatory changes, and qualitative signals a human underwriter
  picks up in direct borrower conversations. None of these are in the feature
  set, and none of them ever will be from financial ratios alone.
- **Small sample size:** with 1,000 loans and ~56 defaults, both models are
  trained on a genuinely small dataset for the number of features and the
  complexity of the problem. Confidence intervals on these metrics are wide;
  a materially larger, real loan book would be needed before either model's
  performance estimate should be treated as stable.

**Conclusion: this model is built to prioritize which loans a human credit
officer reviews first, not to approve, decline, or reprice a loan on its own.**
The Streamlit app's Credit Review Copilot enforces this architecturally — it
has no write access to loan status or covenants, and it flags missing data
rather than estimating it.

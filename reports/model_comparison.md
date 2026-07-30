# Model Comparison

Trained: 2026-07-30

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| roc_auc | 0.7881 | 0.7388 |
| pr_auc | 0.3276 | 0.2846 |
| f1_at_0.5 | 0.2308 | 0.2667 |
| precision_at_0.5 | 0.1406 | 0.25 |
| recall_at_0.5 | 0.6429 | 0.2857 |

**Production model: XGBoost, with a caveat stated plainly.** On this specific
dataset (1,000 loans, ~56 defaults in the training fold), logistic
regression actually has the higher PR-AUC (0.328 vs. 0.285) and higher
recall at the 0.5 threshold (0.643 vs. 0.286). This is a known small-sample
effect: gradient boosting needs more positive examples than this dataset
has to find non-linear patterns a linear model can't already capture, so
on raw performance the simpler model currently wins.

XGBoost is still shipped as the production model in this project, for two
explicit reasons, not because its metrics are better:
1. It powers the SHAP-based per-loan explainability shown in the app's
   Loan Explorer, which is a stated deliverable for the copilot layer.
2. It is expected to overtake logistic regression as the loan tape grows
   past a few thousand loans and default counts rise — which is the
   realistic operating scale for this kind of platform in production.

A live production deployment on a dataset this size should seriously
consider running logistic regression as the primary score and XGBoost as
a secondary/explainability model, or simply logistic regression alone
until enough default history accumulates. Choosing XGBoost here despite
the metrics is a project design decision worth defending in an interview,
not a result to hide.

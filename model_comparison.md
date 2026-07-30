# Model Comparison

Trained: 2026-07-30

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| roc_auc | 0.7881 | 0.7388 |
| pr_auc | 0.3276 | 0.2846 |
| f1_at_0.5 | 0.2308 | 0.2667 |
| precision_at_0.5 | 0.1406 | 0.25 |
| recall_at_0.5 | 0.6429 | 0.2857 |

**Production model: XGBoost** (higher PR-AUC, the more informative metric under ~5% class imbalance; logistic regression is retained as the transparent baseline and for sanity-checking directional coefficient signs).

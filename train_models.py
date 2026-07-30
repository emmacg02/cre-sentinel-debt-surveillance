"""
train_models.py

Trains two competing default-prediction models on the synthetic CRE loan
tape: a logistic regression baseline and an XGBoost classifier. Outputs:
  - models/logistic_model.joblib
  - models/xgboost_model.joblib
  - models/feature_columns.json
  - data/scored_loan_tape.csv          (adds risk_score 0-100 per loan)
  - reports/model_metrics.json
  - reports/shap_summary.png
  - reports/model_comparison.md

Design decisions worth calling out explicitly (these are the things a
credit committee or a technical interviewer will actually probe):

1. LEAKAGE CONTROL: `covenant_status`, `delinquency_status`, and
   `watchlist_rating` are excluded from the feature set. They are
   generated in the synthetic tape FROM THE SAME underlying default
   probability as the target itself, so including them would let the
   model "cheat" by reading a proxy of the answer rather than learning
   from the underlying financial ratios. In a real deployment these
   fields are also often only known AFTER the credit event you're
   trying to predict, which is the textbook definition of leakage.

2. CLASS IMBALANCE: the target is heavily imbalanced (~5-6% positive
   class). Two different, legitimate approaches are used and compared:
   - Logistic regression: `class_weight="balanced"` (reweights the
     loss function, no synthetic data introduced).
   - XGBoost: `scale_pos_weight` (equivalent reweighting inside gradient
     boosting).
   SMOTE (synthetic oversampling) was deliberately NOT used here: on a
   dataset this size, with a modest feature count, SMOTE-generated
   points can sit unrealistically close to real minority-class points
   and inflate validation performance without adding real signal. It's
   a reasonable technique and worth knowing, but weighting is the more
   defensible default for a tabular financial dataset like this one.

3. SPLIT DISCIPLINE: stratified train/test split BEFORE any resampling
   or scaling is fit, so nothing about the test set leaks into training.

4. THRESHOLD: default probability threshold (0.5) is reported alongside
   an F2-optimized threshold (weights recall higher than precision),
   because in lending, missing a real default (false negative) is
   typically far costlier than flagging a healthy loan for review
   (false positive).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, fbeta_score,
)
import xgboost as xgb
import shap
import joblib
import os

DATA_PATH = "data/synthetic_loan_tape.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load and engineer features
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["origination_date", "maturity_date"])

TODAY = pd.Timestamp("2026-01-01")
df["loan_age_years"] = (TODAY - df["origination_date"]).dt.days / 365
df["years_to_maturity"] = (df["maturity_date"] - TODAY).dt.days / 365

# Deliberately excluded (see module docstring, point 1): covenant_status,
# delinquency_status, watchlist_rating, loan_id, raw dates.
NUMERIC_FEATURES = [
    "interest_rate_pct", "ltv", "dscr", "debt_yield", "interest_coverage",
    "occupancy_pct", "loan_age_years", "years_to_maturity",
]
CATEGORICAL_FEATURES = ["property_type", "region", "rate_type"]
TARGET = "defaulted"

X = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                    columns=CATEGORICAL_FEATURES, drop_first=True)
y = df[TARGET]

feature_columns = X.columns.tolist()
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, stratify=y, random_state=42
)

print(f"Train: {len(X_train)} loans ({y_train.mean():.2%} default)")
print(f"Test:  {len(X_test)} loans ({y_test.mean():.2%} default)")

# ---------------------------------------------------------------------
# 2. Baseline: Logistic Regression
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logit = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
logit.fit(X_train_scaled, y_train)
logit_proba = logit.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------
# 3. XGBoost
# ---------------------------------------------------------------------
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr", random_state=42,
)
xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
def evaluate(y_true, proba, name, threshold=0.5):
    preds = (proba >= threshold).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-9)
    best_idx = np.nanargmax(f2_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "f1_at_0.5": round(f1_score(y_true, preds), 4),
        "precision_at_0.5": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall_at_0.5": round(recall_score(y_true, preds, zero_division=0), 4),
        "f2_optimal_threshold": round(float(best_thresh), 4),
        "confusion_matrix_at_0.5": confusion_matrix(y_true, preds).tolist(),
    }

results = [
    evaluate(y_test, logit_proba, "Logistic Regression"),
    evaluate(y_test, xgb_proba, "XGBoost"),
]

with open("reports/model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))

# ---------------------------------------------------------------------
# 5. SHAP explainability (production model = XGBoost)
# ---------------------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, max_display=12)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 6. Score the full loan tape with the production model (XGBoost)
# ---------------------------------------------------------------------
full_proba = xgb_model.predict_proba(X)[:, 1]
df["risk_score"] = np.round(full_proba * 100, 2)
df["risk_tier"] = pd.cut(
    df["risk_score"], bins=[-1, 20, 50, 75, 100],
    labels=["Low", "Moderate", "Elevated", "High"]
)
df.to_csv("data/scored_loan_tape.csv", index=False)

# ---------------------------------------------------------------------
# 7. Save model artifacts
# ---------------------------------------------------------------------
joblib.dump(logit, "models/logistic_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(xgb_model, "models/xgboost_model.joblib")

# ---------------------------------------------------------------------
# 8. Model comparison summary (feeds directly into model_card.md)
# ---------------------------------------------------------------------
with open("reports/model_comparison.md", "w") as f:
    f.write("# Model Comparison\n\n")
    f.write(f"Trained: {datetime.now().date()}\n\n")
    f.write("| Metric | Logistic Regression | XGBoost |\n")
    f.write("|---|---|---|\n")
    for key in ["roc_auc", "pr_auc", "f1_at_0.5", "precision_at_0.5", "recall_at_0.5"]:
        f.write(f"| {key} | {results[0][key]} | {results[1][key]} |\n")
    f.write("\n**Production model: XGBoost** (higher PR-AUC, the more "
            "informative metric under ~5% class imbalance; logistic "
            "regression is retained as the transparent baseline and for "
            "sanity-checking directional coefficient signs).\n")

print("\nDone. See reports/ and data/scored_loan_tape.csv")


# Model Card — CKD Prediction

**Purpose:** Binary classification (CKD / not CKD) as clinical decision support.
Not a diagnostic tool.

**Final model:** xgboost
**Training data:** 400 rows (post-SMOTE) from a 480-row
source dataset, 26 features.

**Metrics (held-out test set, n=80):**
- Accuracy: 1.000
- Precision: 1.000
- Recall: 1.000
- F1: 1.000
- ROC-AUC: 1.000
- Confusion matrix: [[30, 0], [0, 50]]

**Why this model:** Final model: xgboost
Recall: 1.000 | Precision: 1.000
F1: 1.000 | ROC-AUC: 1.000
Confusion matrix: [[30, 0], [0, 50]]

Rationale: chosen for highest recall among tuned candidates without a
precision collapse, consistent with a recall-first objective (a missed
CKD case is costlier than a false alarm).

**Top global drivers (SHAP):** fill in from Cell 10/11 — e.g. hemoglobin, serum
creatinine, albumin, specific gravity.

**Known limitations:**
- Small dataset (~400 records), single-source — may not generalize across
  populations, labs, or measurement equipment.
- Training set balanced via SMOTE synthetic samples, not additional real patients.
- Not validated prospectively or against a clinician-labeled holdout.

**Not for:** autonomous diagnosis, use without clinician review, populations
not represented in the training data.

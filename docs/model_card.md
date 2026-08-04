# Model Card -- CKD Prediction

**Purpose:** Binary classification (CKD / not CKD) as clinical decision support.
Not a diagnostic tool.

**Final model:** xgboost
**Training data:** 400 total source records (320 train / 80 test,
stratified split); training set balanced to a larger size via SMOTE
before fitting. 26 features.

**Metrics (held-out test set, n=80):**
- Accuracy: 0.975
- Precision: 0.980
- Recall: 0.980
- F1: 0.980
- ROC-AUC: 0.999
- Confusion matrix: [[29, 1], [1, 49]]

**Why this model:** Final model: xgboost
Recall: 0.980 | Precision: 0.980
F1: 0.980 | ROC-AUC: 0.999
Confusion matrix: [[29, 1], [1, 49]]

Rationale: chosen for the strongest recall among tuned candidates without a precision collapse, consistent with a recall-first objective (a missed CKD case is costlier than a false alarm).

**Top global drivers (SHAP):** see notebook plots for the ranked list; the most
correlated raw feature is hemo (see limitations below).

**Known limitations:**
- This dataset is highly separable on hemo
  (correlation -0.726 with the target), which likely explains the
  near-perfect test performance above. Results may not generalize to borderline
  or early-stage CKD cases where this value is less extreme, or to
  populations/labs where this feature behaves differently.
- Small dataset (~480 records), single-source -- may not generalize across
  populations, labs, or measurement equipment.
- Training set balanced via SMOTE synthetic samples, not additional real patients.
- Not validated prospectively or against a clinician-labeled holdout.

**Not for:** autonomous diagnosis, use without clinician review, populations
not represented in the training data.

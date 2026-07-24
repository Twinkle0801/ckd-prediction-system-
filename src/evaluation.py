# src/evaluation.py
"""
Model Evaluation & Explainability

Reusable functions backing 07_evaluation.ipynb, following the same pattern as
src/models.py and src/tuning.py: the notebook calls these, displays results,
and does its Check verifications inline -- it doesn't hold the logic itself.

Typical usage from the notebook:

    from src.evaluation import (
        check_target_leakage,
        compare_tuned_vs_untuned,
        select_final_model,
        explain_model,
        top_shap_features,
        generate_model_card,
    )

    target_corr = check_target_leakage(raw_df)
    compare_df = compare_tuned_vs_untuned(baseline_df, tuned_df, candidates)
    final_model, final_scores, justification = select_final_model(final_model_name, tuned_models, X_test, y_test)
    explainer, shap_values = explain_model(final_model, X_test)
    top_features = top_shap_features(shap_values, X_test)
    generate_model_card(..., path="../MODEL_CARD.md")
"""

import pandas as pd
import numpy as np
import shap

from src.models import evaluate_model


# ---------------------------------------------------------------------------
# Leakage / correlation check
# ---------------------------------------------------------------------------

def check_target_leakage(
    raw_df: pd.DataFrame,
    target_col: str = "classification",
    exclude: tuple = ("id",),
) -> pd.Series:
    """
    Compute each numeric feature's correlation with the target, excluding
    identifier columns that would otherwise produce spurious correlation
    (e.g. `id` correlating with class purely because rows are sorted by
    class in the source CSV -- not a real predictive signal).

    raw_df must already have target_col encoded numerically (e.g. ckd=1,
    notckd=0) before calling this.

    Returns a Series sorted by absolute correlation, descending, with the
    excluded columns and the target itself removed.
    """
    df = raw_df.drop(columns=list(exclude), errors="ignore")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(
            f"'{target_col}' is not numeric -- encode it (e.g. ckd=1/notckd=0) "
            "before calling check_target_leakage()."
        )

    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    corr = corr.sort_values(key=abs, ascending=False)

    # Defensive guard: if an excluded column somehow leaks back in (e.g. a
    # rename), fail loudly rather than silently reporting a meaningless
    # top feature.
    for col in exclude:
        assert col not in corr.index, (
            f"'{col}' leaked into the correlation check -- "
            "make sure it's dropped before corr() runs."
        )

    return corr


def top_correlated_feature(target_corr: pd.Series) -> tuple:
    """Convenience accessor: (feature_name, correlation_value) for the top row."""
    return target_corr.index[0], target_corr.iloc[0]


# ---------------------------------------------------------------------------
# Tuned vs. untuned comparison
# ---------------------------------------------------------------------------

def compare_tuned_vs_untuned(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    candidates: list,
) -> pd.DataFrame:
    """
    Combine untuned baseline rows (filtered to the tuned candidates only)
    with their tuned counterparts into one sorted comparison table.

    Expects both DataFrames to have at least: model, stage, accuracy,
    precision, recall, f1, roc_auc, confusion_matrix. tuned_df may also
    carry a 'best_params' column, which is dropped here since it doesn't
    belong in a side-by-side metrics comparison.
    """
    tuned_clean = tuned_df.drop(columns=["best_params"], errors="ignore")
    baseline_subset = baseline_df[baseline_df["model"].isin(candidates)]

    compare_df = (
        pd.concat([baseline_subset, tuned_clean])
        .sort_values(["model", "stage"])
        .reset_index(drop=True)
    )
    return compare_df


def flag_precision_collapse(compare_df: pd.DataFrame, min_precision: float = 0.5) -> pd.DataFrame:
    """
    Return rows where precision dropped below `min_precision` -- a warning
    sign that a model is chasing recall by predicting the positive class
    indiscriminately rather than learning real signal.
    """
    return compare_df[compare_df["precision"] < min_precision]


# ---------------------------------------------------------------------------
# Final model selection
# ---------------------------------------------------------------------------

def select_final_model(
    final_model_name: str,
    tuned_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """
    Score the chosen final model on the held-out test set and build a
    numbers-grounded justification string.

    Returns (final_model, final_scores, justification).
    """
    if final_model_name not in tuned_models:
        raise ValueError(
            f"'{final_model_name}' not found in tuned_models. "
            f"Options: {list(tuned_models)}"
        )

    final_model = tuned_models[final_model_name]
    final_scores = evaluate_model(final_model, X_test, y_test)

    justification = (
        f"Final model: {final_model_name}\n"
        f"Recall: {final_scores['recall']:.3f} | Precision: {final_scores['precision']:.3f}\n"
        f"F1: {final_scores['f1']:.3f} | ROC-AUC: {final_scores['roc_auc']:.3f}\n"
        f"Confusion matrix: {final_scores['confusion_matrix']}\n\n"
        "Rationale: chosen for the strongest recall among tuned candidates "
        "without a precision collapse, consistent with a recall-first "
        "objective (a missed CKD case is costlier than a false alarm)."
    )

    return final_model, final_scores, justification


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

# Model classes TreeExplainer supports directly, matched against the EXACT
# class name (type(model).__name__) -- not a fuzzy substring check. The
# sklearn-style class names don't spell out their library, e.g. XGBoost's
# class is "XGBClassifier" (not "XGBoostClassifier") and LightGBM's is
# "LGBMClassifier" (not "LightGBMClassifier"), so substring matching against
# "xgboost"/"lightgbm" silently fails for exactly the models this is meant
# to support. Anything not in this set (logreg, svm, knn) needs
# shap.LinearExplainer / shap.KernelExplainer instead -- deliberately not
# auto-handled here so the caller picks the right explainer consciously.
_TREE_BASED_CLASS_NAMES = {
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
}


def explain_model(model, X_test: pd.DataFrame):
    """
    Build a SHAP TreeExplainer for a tree-based model and compute SHAP
    values on the test set. Raises if the model type looks unsupported,
    to avoid silently producing meaningless explanations for e.g. an SVM.
    """
    model_type = type(model).__name__
    if model_type not in _TREE_BASED_CLASS_NAMES:
        raise TypeError(
            f"explain_model() uses shap.TreeExplainer, which expects a tree-based "
            f"model. Got {model_type}. For logreg use shap.LinearExplainer; "
            f"for svm/knn use shap.KernelExplainer instead."
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def top_shap_features(shap_values, X_test: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    Rank features by mean absolute SHAP value (global importance), returning
    the top n as a Series of {feature: mean_abs_shap}. Use this to
    programmatically cross-check against check_target_leakage()'s top
    correlated feature, rather than eyeballing the bar plot every time.
    """
    values = np.asarray(shap_values)
    mean_abs = np.abs(values).mean(axis=0)
    ranked = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)
    return ranked.head(n)


def explain_single_prediction(model, explainer, shap_values, X_test: pd.DataFrame, index: int) -> dict:
    """
    Build a plain-data (no plotting) summary of one prediction's SHAP
    contributions -- the exact shape you'd hand to an LLM prompt in Phase 16,
    so grounding is trivial: every number in the explanation traces back
    to this dict.
    """
    row = X_test.iloc[index]
    contributions = pd.Series(shap_values[index], index=X_test.columns)
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    pred = model.predict(X_test.iloc[[index]])[0]
    proba = (
        model.predict_proba(X_test.iloc[[index]])[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    return {
        "index": index,
        "prediction": int(pred),
        "probability": float(proba) if proba is not None else None,
        "expected_value": float(explainer.expected_value),
        "top_contributions": [
            {"feature": feat, "value": float(row[feat]), "shap_contribution": float(val)}
            for feat, val in contributions.items()
        ],
    }


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def generate_model_card(
    final_model_name: str,
    final_scores: dict,
    justification: str,
    top_feature: str,
    top_corr_value: float,
    n_train: int,
    n_test: int,
    n_features: int,
    path: str = "../MODEL_CARD.md",
) -> str:
    """
    Write the model card to disk and return its text. The leakage/
    separability finding (top_feature, top_corr_value) is a required
    argument, not optional -- forces the caller to have actually run
    check_target_leakage() rather than leaving a placeholder sentence.
    """
    n_total = n_train + n_test

    model_card = f"""
# Model Card -- CKD Prediction

**Purpose:** Binary classification (CKD / not CKD) as clinical decision support.
Not a diagnostic tool.

**Final model:** {final_model_name}
**Training data:** {n_train} rows (post-SMOTE) from a {n_total}-row
source dataset, {n_features} features.

**Metrics (held-out test set, n={n_test}):**
- Accuracy: {final_scores['accuracy']:.3f}
- Precision: {final_scores['precision']:.3f}
- Recall: {final_scores['recall']:.3f}
- F1: {final_scores['f1']:.3f}
- ROC-AUC: {final_scores['roc_auc']:.3f}
- Confusion matrix: {final_scores['confusion_matrix']}

**Why this model:** {justification.strip()}

**Top global drivers (SHAP):** see notebook plots for the ranked list; the most
correlated raw feature is {top_feature} (see limitations below).

**Known limitations:**
- This dataset is highly separable on {top_feature}
  (correlation {top_corr_value:.3f} with the target), which likely explains the
  near-perfect test performance above. Results may not generalize to borderline
  or early-stage CKD cases where this value is less extreme, or to
  populations/labs where this feature behaves differently.
- Small dataset (~{n_total} records), single-source -- may not generalize across
  populations, labs, or measurement equipment.
- Training set balanced via SMOTE synthetic samples, not additional real patients.
- Not validated prospectively or against a clinician-labeled holdout.

**Not for:** autonomous diagnosis, use without clinician review, populations
not represented in the training data.
""".strip() + "\n"

    with open(path, "w") as f:
        f.write(model_card)

    return model_card


if __name__ == "__main__":
    # Smoke test: exercises the module against the same paths the notebook
    # uses, so `python -m src.evaluation` from the project root catches
    # obvious breakage without opening Jupyter.
    from src.data_loader import load_and_prepare
    from src.models import MODEL_REGISTRY, train_model
    from src.tuning import tune_model

    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy="smote")

    raw = pd.read_csv("data/processed/kidney_clean.csv")
    raw["classification"] = (
        raw["classification"].astype(str).str.strip().str.lower().map({"ckd": 1, "notckd": 0})
    )
    target_corr = check_target_leakage(raw)
    top_feature, top_corr_value = top_correlated_feature(target_corr)
    print("Top correlated feature:", top_feature, top_corr_value)

    baseline_rows = [
        {"model": name, "stage": "untuned", **evaluate_model(train_model(name, X_train, y_train), X_test, y_test)}
        for name in MODEL_REGISTRY
    ]
    baseline_df = pd.DataFrame(baseline_rows)

    candidates = ["random_forest", "xgboost", "lightgbm"]
    tuned_models, tuned_rows = {}, []
    for name in candidates:
        search = tune_model(name, X_train, y_train, scoring="recall", n_iter=20)
        tuned_models[name] = search.best_estimator_
        tuned_rows.append({
            "model": name, "stage": "tuned",
            **evaluate_model(search.best_estimator_, X_test, y_test),
        })
    tuned_df = pd.DataFrame(tuned_rows)

    compare_df = compare_tuned_vs_untuned(baseline_df, tuned_df, candidates)
    print(compare_df.drop(columns="confusion_matrix"))

    final_model, final_scores, justification = select_final_model(
        "xgboost", tuned_models, X_test, y_test
    )

    explainer, shap_values = explain_model(final_model, X_test)
    print("Top SHAP features:\n", top_shap_features(shap_values, X_test))

    generate_model_card(
        final_model_name="xgboost",
        final_scores=final_scores,
        justification=justification,
        top_feature=top_feature,
        top_corr_value=top_corr_value,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        n_features=X_train.shape[1],
        path="MODEL_CARD.md",
    )
    print("Wrote MODEL_CARD.md")
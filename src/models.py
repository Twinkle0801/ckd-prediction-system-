# src/models.py
"""
Day 6 — Model Building

Common training/evaluation interface so every classifier is trained and
scored the same way. Feed this the outputs of src/data_loader.py:

    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy="smote")
    results_df, fitted_models = train_and_evaluate_all(X_train, y_train, X_test, y_test)

IMPORTANT: X_train/y_train here should already be the SMOTE'd (or
undersampled / raw) training set from Day 5. X_test/y_test must be the
untouched holdout set — never resampled.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model registry — add/remove models here, nothing else needs to change.
# These are TEMPLATE instances — never fit directly. train_model() clones
# a fresh copy from this registry every time it's called (see FIX below).
# ---------------------------------------------------------------------------

def _build_registry() -> dict:
    registry = {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True, random_state=RANDOM_STATE),  # probability=True needed for ROC-AUC
    }
    if _HAS_XGB:
        registry["xgboost"] = XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE
        )
    if _HAS_LGBM:
        registry["lightgbm"] = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
    return registry


MODEL_REGISTRY = _build_registry()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit one model by name from MODEL_REGISTRY. Returns a freshly fitted model.

    FIX: previously this called .fit() directly on the shared object living
    in MODEL_REGISTRY, so retraining the same model name later (e.g. under a
    different imbalance strategy in the same session) silently overwrote the
    earlier fitted model out from under any variable still referencing it.
    clone() gives every call an independent, unfitted copy first.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_REGISTRY)}")
    model = clone(MODEL_REGISTRY[name])
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Score a fitted model on held-out data. Recall-first project, so recall
    and F1 matter more than raw accuracy — a missed CKD case is the costly
    error, not a false alarm.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # ROC-AUC needs predicted probabilities, not hard labels
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


# ---------------------------------------------------------------------------
# Train + evaluate every registered model
# ---------------------------------------------------------------------------

def train_and_evaluate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Loop over MODEL_REGISTRY, train each model on (X_train, y_train),
    evaluate each on (X_test, y_test), and return one comparison table
    sorted by recall (then F1) descending.
    """
    rows = []
    fitted_models = {}

    for name in MODEL_REGISTRY:
        model = train_model(name, X_train, y_train)
        fitted_models[name] = model
        scores = evaluate_model(model, X_test, y_test)
        row = {"model": name, **scores}
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["recall", "f1"], ascending=False
    ).reset_index(drop=True)

    return results_df, fitted_models


if __name__ == "__main__":
    from src.data_loader import load_and_prepare

    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy="smote")
    results_df, fitted_models = train_and_evaluate_all(X_train, y_train, X_test, y_test)
    print(results_df.drop(columns="confusion_matrix"))
    results_df.to_csv("data/processed/model_comparison.csv", index=False)
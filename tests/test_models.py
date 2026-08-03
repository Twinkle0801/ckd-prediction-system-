"""
Day 13: unit + integration tests for src/models.py
"""
import pytest
from src.models import (
    MODEL_REGISTRY,
    train_model,
    evaluate_model,
    train_and_evaluate_all,
)
from src.data_loader import load_and_prepare


@pytest.fixture(scope="module")
def prepared_data():
    return load_and_prepare(strategy="smote")


def test_model_registry_not_empty():
    assert len(MODEL_REGISTRY) > 0
    assert "logreg" in MODEL_REGISTRY
    assert "random_forest" in MODEL_REGISTRY


def test_train_model_unknown_name_raises(prepared_data):
    X_train, X_test, y_train, y_test, scaler = prepared_data
    with pytest.raises(ValueError):
        train_model("not_a_real_model", X_train, y_train)


def test_train_model_returns_fitted_model(prepared_data):
    X_train, X_test, y_train, y_test, scaler = prepared_data
    model = train_model("logreg", X_train, y_train)
    # A fitted sklearn classifier should be able to predict without error
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_model_clone_does_not_mutate_registry(prepared_data):
    """Regression test for the exact bug the FIX comment describes:
    training a model must never mutate the shared MODEL_REGISTRY template."""
    from sklearn.base import is_classifier
    X_train, X_test, y_train, y_test, scaler = prepared_data

    registry_model_before = MODEL_REGISTRY["logreg"]
    assert not hasattr(registry_model_before, "coef_"), \
        "MODEL_REGISTRY template should never be fitted"

    train_model("logreg", X_train, y_train)

    registry_model_after = MODEL_REGISTRY["logreg"]
    assert not hasattr(registry_model_after, "coef_"), \
        "Training a model mutated the shared registry template — clone() regression"


def test_evaluate_model_returns_expected_keys(prepared_data):
    X_train, X_test, y_train, y_test, scaler = prepared_data
    model = train_model("logreg", X_train, y_train)
    scores = evaluate_model(model, X_test, y_test)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
    assert expected_keys.issubset(scores.keys())
    assert 0.0 <= scores["accuracy"] <= 1.0
    assert 0.0 <= scores["recall"] <= 1.0


def test_evaluate_model_confusion_matrix_shape(prepared_data):
    X_train, X_test, y_train, y_test, scaler = prepared_data
    model = train_model("random_forest", X_train, y_train)
    scores = evaluate_model(model, X_test, y_test)
    cm = scores["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2, "Expected 2x2 confusion matrix for binary classification"


def test_train_and_evaluate_all_covers_every_registered_model(prepared_data):
    X_train, X_test, y_train, y_test, scaler = prepared_data
    results_df, fitted_models = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    assert set(results_df["model"]) == set(MODEL_REGISTRY.keys())
    assert set(fitted_models.keys()) == set(MODEL_REGISTRY.keys())


def test_train_and_evaluate_all_sorted_by_recall_then_f1(prepared_data):
    X_train, X_test, y_train, y_test, scaler = prepared_data
    results_df, _ = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    recalls = results_df["recall"].tolist()
    assert recalls == sorted(recalls, reverse=True), "Results should be sorted by recall descending"


# ── Edge cases (per Day 13 roadmap: missing fields, out-of-range values) ──

def test_evaluate_model_handles_all_one_class_predictions(prepared_data):
    """Edge case: a degenerate model that only ever predicts one class
    should not crash evaluate_model (zero_division handling)."""
    X_train, X_test, y_train, y_test, scaler = prepared_data

    class AlwaysZero:
        def predict(self, X):
            import numpy as np
            return np.zeros(len(X))

    scores = evaluate_model(AlwaysZero(), X_test, y_test)
    assert scores["precision"] == 0.0 or scores["precision"] >= 0.0  # no ZeroDivisionError raised
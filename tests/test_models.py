# tests/test_models.py
import numpy as np
import pandas as pd
import pytest
from src.models import (
    MODEL_REGISTRY, train_model, evaluate_model, train_and_evaluate_all,
)

def test_model_registry_contains_expected_core_models():
    # xgboost/lightgbm are conditional on import success, so only assert the
    # always-present five
    for name in ["logreg", "decision_tree", "random_forest", "knn", "svm"]:
        assert name in MODEL_REGISTRY

def test_train_model_unknown_name_raises():
    with pytest.raises(ValueError):
        train_model("not_a_real_model", pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))

def test_train_model_clones_not_mutates_registry():
    """Regression test for the exact bug the FIX comment describes: training
    twice under the same name must not silently share state via the registry."""
    X = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    y = pd.Series([0, 1] * 10)
    model_1 = train_model("logreg", X, y)
    assert not hasattr(MODEL_REGISTRY["logreg"], "coef_")  # template stays unfitted
    model_2 = train_model("logreg", X, y)
    assert model_1 is not model_2  # two independent fitted objects

def test_evaluate_model_perfect_predictions_give_perfect_scores():
    class DummyPerfectModel:
        def fit(self, X, y): self.y_ = y; return self
        def predict(self, X): return self.y_
        def predict_proba(self, X):
            return np.column_stack([1 - self.y_, self.y_])
    y_test = pd.Series([1, 0, 1, 0, 1])
    model = DummyPerfectModel().fit(None, y_test)
    scores = evaluate_model(model, X_test=np.zeros((5, 1)), y_test=y_test)
    assert scores["accuracy"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["roc_auc"] == 1.0

def test_evaluate_model_known_confusion_matrix():
    # 3 TP, 1 FN, 0 FP, 1 TN -> recall = 3/4 = 0.75
    y_test = pd.Series([1, 1, 1, 1, 0])
    class OneMissModel:
        def predict(self, X): return np.array([1, 1, 1, 0, 0])
        def predict_proba(self, X): return np.column_stack([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
    scores = evaluate_model(OneMissModel(), X_test=np.zeros((5, 1)), y_test=y_test)
    assert round(scores["recall"], 2) == 0.75
    assert scores["confusion_matrix"] == [[1, 0], [1, 3]]  # [[TN,FP],[FN,TP]]

def test_evaluate_model_handles_no_predict_proba():
    """A model without predict_proba should degrade to roc_auc=None, not crash."""
    class NoProbaModel:
        def predict(self, X): return np.array([1, 0, 1])
    scores = evaluate_model(NoProbaModel(), X_test=np.zeros((3, 1)), y_test=pd.Series([1, 0, 1]))
    assert scores["roc_auc"] is None

def test_train_and_evaluate_all_returns_sorted_by_recall():
    X_train = pd.DataFrame(np.random.rand(40, 4), columns=list("abcd"))
    y_train = pd.Series([0, 1] * 20)
    X_test = pd.DataFrame(np.random.rand(10, 4), columns=list("abcd"))
    y_test = pd.Series([0, 1] * 5)
    results_df, fitted_models = train_and_evaluate_all(X_train, y_train, X_test, y_test)
    assert list(results_df["recall"]) == sorted(results_df["recall"], reverse=True)
    assert set(fitted_models.keys()) == set(MODEL_REGISTRY.keys())
# tests/test_tuning.py
import numpy as np
from src.tuning import tune_model, PARAM_GRIDS
from src.models import MODEL_REGISTRY  # confirmed name — dict of template estimators

def test_tune_model_returns_params_within_grid():
    X = np.random.rand(60, 5)
    y = np.array([0, 1] * 30)
    search = tune_model("logreg", X, y, cv_splits=3, n_iter=5)
    assert search.best_params_["C"] in PARAM_GRIDS["logreg"]["C"]
    assert search.best_params_["solver"] in PARAM_GRIDS["logreg"]["solver"]

def test_tune_model_is_fitted():
    X = np.random.rand(60, 5)
    y = np.array([0, 1] * 30)
    search = tune_model("decision_tree", X, y, cv_splits=3, n_iter=5)
    assert hasattr(search.best_estimator_, "predict")
    preds = search.best_estimator_.predict(X)
    assert len(preds) == len(y)

def test_all_registered_models_have_param_grids():
    """Every model in MODEL_REGISTRY must have a matching entry in PARAM_GRIDS,
    or tune_model() will KeyError on that model name at runtime."""
    for name in MODEL_REGISTRY:
        assert name in PARAM_GRIDS, f"{name} is registered but has no PARAM_GRIDS entry"
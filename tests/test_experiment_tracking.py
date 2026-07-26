# tests/test_experiment_tracking.py
import hashlib
import numpy as np
import pandas as pd
import pytest
import mlflow
from mlflow.tracking import MlflowClient

from src.experiment_tracking import (
    setup_tracking, log_baseline_runs, log_tuned_runs, tag_dataset_snapshot,
    get_runs_df, assert_no_duplicate_runs, get_best_run,
    promote_to_production, verify_registered_run_id, load_production_model,
)


@pytest.fixture
def tracking_env(tmp_path):
    """
    Fresh, isolated MLflow tracking store per test -- a real sqlite file in
    tmp_path, NEVER the project's actual mlflow.db. Experiment name is also
    unique per test so tests never see each other's runs.
    """
    db_path = tmp_path / "test_mlflow.db"
    experiment_name = f"test-ckd-{np.random.randint(1_000_000)}"
    setup_tracking(tracking_uri=f"sqlite:///{db_path}", experiment_name=experiment_name)
    return experiment_name


@pytest.fixture
def small_data():
    X_train = pd.DataFrame(np.random.rand(40, 3), columns=list("abc"))
    y_train = pd.Series([0, 1] * 20)
    X_test = pd.DataFrame(np.random.rand(10, 3), columns=list("abc"))
    y_test = pd.Series([0, 1] * 5)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def tiny_model_registry():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }


def test_setup_tracking_creates_experiment(tracking_env):
    experiment = mlflow.get_experiment_by_name(tracking_env)
    assert experiment is not None


def test_log_baseline_runs_logs_one_run_per_model(tracking_env, small_data, tiny_model_registry):
    X_train, y_train, X_test, y_test = small_data
    count = log_baseline_runs(tiny_model_registry, X_train, y_train, X_test, y_test)
    assert count == len(tiny_model_registry)

    runs_df = get_runs_df(tracking_env)
    assert len(runs_df) == len(tiny_model_registry)
    assert set(runs_df["tags.model_family"]) == set(tiny_model_registry.keys())
    assert set(runs_df["tags.stage"]) == {"untuned"}


def test_log_baseline_runs_called_twice_creates_duplicates(tracking_env, small_data, tiny_model_registry):
    """
    Documents the exact DANGEROUS behavior the module's docstring warns
    about: calling log_baseline_runs twice does NOT dedupe -- it doubles
    the run count. This locks that behavior in so nobody accidentally
    'fixes' it without noticing assert_no_duplicate_runs depends on it.
    """
    X_train, y_train, X_test, y_test = small_data
    log_baseline_runs(tiny_model_registry, X_train, y_train, X_test, y_test)
    log_baseline_runs(tiny_model_registry, X_train, y_train, X_test, y_test)

    runs_df = get_runs_df(tracking_env)
    assert len(runs_df) == 2 * len(tiny_model_registry)

    with pytest.raises(ValueError, match="Duplicate run names"):
        assert_no_duplicate_runs(runs_df)


def test_log_tuned_runs_logs_expected_params_and_returns_fitted_models(tracking_env, small_data):
    X_train, y_train, X_test, y_test = small_data
    tuned_models = log_tuned_runs(["logreg"], X_train, y_train, X_test, y_test, n_iter=3)

    assert "logreg" in tuned_models
    assert hasattr(tuned_models["logreg"], "predict")

    runs_df = get_runs_df(tracking_env)
    assert len(runs_df) == 1
    assert runs_df.iloc[0]["tags.stage"] == "tuned"
    best_param_cols = [c for c in runs_df.columns if c.startswith("params.best_")]
    assert len(best_param_cols) > 0


def test_get_best_run_finds_matching_row(tracking_env, small_data, tiny_model_registry):
    X_train, y_train, X_test, y_test = small_data
    log_baseline_runs(tiny_model_registry, X_train, y_train, X_test, y_test)
    runs_df = get_runs_df(tracking_env)
    best_run = get_best_run(runs_df, model_family="logreg", stage="untuned")
    assert best_run["tags.model_family"] == "logreg"


def test_get_best_run_raises_when_no_match(tracking_env, small_data, tiny_model_registry):
    X_train, y_train, X_test, y_test = small_data
    log_baseline_runs(tiny_model_registry, X_train, y_train, X_test, y_test)
    runs_df = get_runs_df(tracking_env)
    with pytest.raises(ValueError, match="No run found"):
        get_best_run(runs_df, model_family="xgboost", stage="tuned")  # never logged


def test_tag_dataset_snapshot_returns_correct_hash_and_tags_run(
    tracking_env, small_data, tiny_model_registry, tmp_path
):
    X_train, y_train, X_test, y_test = small_data
    log_baseline_runs({"logreg": tiny_model_registry["logreg"]}, X_train, y_train, X_test, y_test)
    runs_df = get_runs_df(tracking_env)
    run_id = runs_df.iloc[0]["run_id"]

    dataset_path = tmp_path / "snapshot.csv"
    X_train.to_csv(dataset_path, index=False)
    expected_hash = hashlib.md5(dataset_path.read_bytes()).hexdigest()

    returned_hash = tag_dataset_snapshot(run_id, str(dataset_path))
    assert returned_hash == expected_hash

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags["dataset_hash"] == expected_hash


def test_promote_to_production_and_load_round_trip(tracking_env, small_data, tiny_model_registry):
    X_train, y_train, X_test, y_test = small_data
    log_baseline_runs({"logreg": tiny_model_registry["logreg"]}, X_train, y_train, X_test, y_test)
    runs_df = get_runs_df(tracking_env)
    run_id = runs_df.iloc[0]["run_id"]

    registered_name = f"test-model-{np.random.randint(1_000_000)}"
    registered = promote_to_production(registered_name, run_id)

    assert verify_registered_run_id(registered_name, registered.version, run_id) is True

    production_model = load_production_model(registered_name)
    assert hasattr(production_model, "predict")
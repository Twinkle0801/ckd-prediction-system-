# src/experiment_tracking.py
"""
Experiment Tracking (MLflow)
 
Reusable functions backing 08_experiment_tracking.ipynb, following the same
pattern as src/models.py, src/tuning.py, and src/evaluation.py: the notebook
calls these, displays results, and does its Check verifications inline -- it
doesn't hold the logic itself.
 
IMPORTANT -- read before calling anything in this file:
log_baseline_runs() and log_tuned_runs() each create NEW MLflow runs every
time they're called -- MLflow runs are append-only, there is no "overwrite".
Calling either of these more than once against the same experiment WILL
produce duplicate runs (this is exactly what happened during manual notebook
debugging on Day 9 and required a full run cleanup to fix). Only call them
when you actually intend to log a fresh round of training -- never "just to
check something works."
 
load_production_model() and get_best_run() are read-only and safe to call
as often as you like.
 
Typical usage from the notebook:
 
    from src.experiment_tracking import (
        setup_tracking,
        log_baseline_runs,
        log_tuned_runs,
        get_best_run,
        promote_to_production,
        load_production_model,
        tag_dataset_snapshot,
    )
 
    setup_tracking()                                   # safe, idempotent
    log_baseline_runs(MODEL_REGISTRY, X_train, y_train, X_test, y_test)   # DANGEROUS: only once
    tuned_models = log_tuned_runs(candidates, X_train, y_train, X_test, y_test)  # DANGEROUS: only once
 
    runs_df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])       # safe, read-only
    best_run = get_best_run(runs_df, "xgboost", "tuned")                   # safe, read-only
    promote_to_production("ckd_prediction_model", best_run["run_id"])      # only when promoting a new run
 
    production_model = load_production_model("ckd_prediction_model")      # safe, read-only
"""
 
import hashlib
 
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
 
from src.models import evaluate_model, train_model
from src.tuning import tune_model
 
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "ckd-prediction"
 
 
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
 
def setup_tracking(tracking_uri: str = TRACKING_URI, experiment_name: str = EXPERIMENT_NAME) -> None:
    """
    Point mlflow at the SQLite backend and select the experiment.
 
    Uses a SQLite store, not plain file storage ("file:../mlruns") -- current
    mlflow versions put file-based tracking in maintenance mode and raise an
    MlflowException on set_experiment() if you try to use it.
 
    Safe to call repeatedly -- setting the tracking URI / experiment multiple
    times does not create duplicate experiments or runs by itself.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
 
 
# ---------------------------------------------------------------------------
# Logging -- DANGEROUS to call more than once per training round
# ---------------------------------------------------------------------------
 
def log_baseline_runs(
    model_registry: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    resampling_label: str = "smote",
) -> int:
    """
    Train and log every model in model_registry as its own untuned MLflow run.
 
    WARNING: each call creates len(model_registry) NEW runs. Calling this
    twice against the same experiment produces duplicates -- there is no
    dedup or overwrite behavior. Only call this once per fresh training round.
 
    Returns the number of runs logged.
    """
    count = 0
    for name in model_registry:
        with mlflow.start_run(run_name=f"{name}_untuned"):
            model = train_model(name, X_train, y_train)
            scores = evaluate_model(model, X_test, y_test)
 
            mlflow.set_tag("stage", "untuned")
            mlflow.set_tag("model_family", name)
 
            mlflow.log_param("model", name)
            mlflow.log_param("resampling", resampling_label)
 
            mlflow.log_metric("accuracy", scores["accuracy"])
            mlflow.log_metric("precision", scores["precision"])
            mlflow.log_metric("recall", scores["recall"])
            mlflow.log_metric("f1", scores["f1"])
            if scores["roc_auc"] is not None:
                mlflow.log_metric("roc_auc", scores["roc_auc"])
 
            # serialization_format="pickle" avoids skops' UntrustedTypesFoundException
            # on XGBoost/LightGBM internals; name= replaces the deprecated artifact_path=
            mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")
            count += 1
 
    return count
 
 
def log_tuned_runs(
    candidates: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "recall",
    n_iter: int = 20,
    resampling_label: str = "smote",
) -> dict:
    """
    Tune and log each candidate model as its own tuned MLflow run, with the
    winning hyperparameters logged as params (prefixed best_).
 
    WARNING: same as log_baseline_runs -- each call creates len(candidates)
    NEW runs. Only call this once per fresh training round.
 
    Returns {model_name: fitted_best_estimator} for use elsewhere in the
    same session (e.g. immediate SHAP explanation) without re-fitting.
    """
    tuned_models = {}
 
    for name in candidates:
        with mlflow.start_run(run_name=f"{name}_tuned"):
            search = tune_model(name, X_train, y_train, scoring=scoring, n_iter=n_iter)
            best_model = search.best_estimator_
            scores = evaluate_model(best_model, X_test, y_test)
            tuned_models[name] = best_model
 
            mlflow.set_tag("stage", "tuned")
            mlflow.set_tag("model_family", name)
 
            mlflow.log_param("model", name)
            mlflow.log_param("resampling", resampling_label)
            for k, v in search.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
 
            mlflow.log_metric("accuracy", scores["accuracy"])
            mlflow.log_metric("precision", scores["precision"])
            mlflow.log_metric("recall", scores["recall"])
            mlflow.log_metric("f1", scores["f1"])
            if scores["roc_auc"] is not None:
                mlflow.log_metric("roc_auc", scores["roc_auc"])
 
            mlflow.sklearn.log_model(best_model, name="model", serialization_format="pickle")
 
    return tuned_models
 
 
def tag_dataset_snapshot(run_id: str, dataset_path: str) -> str:
    """
    Reopen an existing run and attach a dataset hash tag plus a browsable
    snapshot artifact of the exact file used to train it. Safe to call
    repeatedly on the same run_id -- it overwrites the tag/artifact rather
    than creating a new run.
 
    Returns the computed hash.
    """
    with open(dataset_path, "rb") as f:
        data_hash = hashlib.md5(f.read()).hexdigest()
 
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("dataset_hash", data_hash)
        mlflow.log_artifact(dataset_path, artifact_path="dataset_snapshot")
 
    return data_hash
 
 
# ---------------------------------------------------------------------------
# Querying -- safe, read-only
# ---------------------------------------------------------------------------
 
def get_runs_df(experiment_name: str = EXPERIMENT_NAME) -> pd.DataFrame:
    """Fetch all runs for the experiment as a DataFrame. Read-only, safe to call anytime."""
    return mlflow.search_runs(experiment_names=[experiment_name])
 
 
def assert_no_duplicate_runs(runs_df: pd.DataFrame) -> None:
    """
    Raise if any run name appears more than once. Call this before trusting
    get_best_run() -- duplicates silently pick whichever row sorts first,
    which may not be your intended run.
    """
    dup = runs_df["tags.mlflow.runName"].duplicated()
    if dup.any():
        dup_names = runs_df.loc[dup, "tags.mlflow.runName"].unique().tolist()
        raise ValueError(
            f"Duplicate run names found: {dup_names}. Clean up the MLflow "
            "runs table before selecting a best run -- see the Day 9 cleanup "
            "steps (select all, delete, re-log exactly once)."
        )
 
 
def get_best_run(runs_df: pd.DataFrame, model_family: str, stage: str) -> pd.Series:
    """
    Return the run row matching model_family + stage. Raises if none found,
    and raises via assert_no_duplicate_runs if the runs table has duplicates
    -- forces cleanup before silently returning a possibly-wrong row.
    """
    assert_no_duplicate_runs(runs_df)
 
    matches = runs_df[
        (runs_df["tags.model_family"] == model_family) & (runs_df["tags.stage"] == stage)
    ]
    if matches.empty:
        raise ValueError(f"No run found for model_family='{model_family}', stage='{stage}'.")
 
    return matches.iloc[0]
 
 
# ---------------------------------------------------------------------------
# Model registry -- promotion is idempotent-ish (creates a new version each
# time, but never duplicates runs), loading is fully read-only
# ---------------------------------------------------------------------------
 
def promote_to_production(registered_model_name: str, run_id: str, artifact_path: str = "model"):
    """
    Register the model from run_id and alias it "production". Creates a new
    registry version each call (v1, v2, ...) -- this does NOT create
    duplicate training runs, only a new registry entry, so it's safe to
    re-run if you genuinely want to promote a different/updated run later.
 
    Returns the registered ModelVersion object.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri, registered_model_name)
 
    client = MlflowClient()
    client.set_registered_model_alias(registered_model_name, "production", registered.version)
 
    return registered
 
 
def verify_registered_run_id(registered_model_name: str, version: str, expected_run_id: str) -> bool:
    """
    Confirm a registered model version actually traces back to the run_id
    you expect. mlflow occasionally resolves registration through its
    internal 'Logged Models' indirection (models:/m-...) and emits a warning
    rather than an error when the requested artifact path isn't found --
    this check catches a genuinely wrong registration rather than trusting
    that warning was harmless.
    """
    client = MlflowClient()
    version_info = client.get_model_version(registered_model_name, version)
    return version_info.run_id == expected_run_id
 
 
def load_production_model(registered_model_name: str):
    """
    Load the model currently aliased "production". This is the function
    Day 10 (Save the Model) and Day 12 (API) should import and call at
    startup -- fully read-only, safe to call as often as needed.
    """
    return mlflow.sklearn.load_model(f"models:/{registered_model_name}@production")
 
 
if __name__ == "__main__":
    # Smoke test: READ-ONLY only. Deliberately does NOT call log_baseline_runs
    # or log_tuned_runs here -- re-running those against an existing experiment
    # would duplicate runs, which is exactly the mistake this module's
    # docstring warns against. This only exercises setup + querying + loading
    # against whatever is ALREADY logged in mlflow.db.
    setup_tracking()
 
    runs_df = get_runs_df()
    print("Total runs found:", len(runs_df))
    assert_no_duplicate_runs(runs_df)
    print("No duplicate run names -- OK.")
 
    best_run = get_best_run(runs_df, model_family="xgboost", stage="tuned")
    print("Best xgboost_tuned run_id:", best_run["run_id"])
 
    production_model = load_production_model("ckd_prediction_model")
    print("Loaded production model:", type(production_model).__name__)
 
    matches_expected = verify_registered_run_id(
        "ckd_prediction_model", "1", best_run["run_id"]
    )
    print("Registered v1 matches best xgboost_tuned run_id:", matches_expected)
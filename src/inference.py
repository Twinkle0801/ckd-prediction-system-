# src/inference.py
"""
Save the Model
 
Reusable inference functions backing 09_save_model.ipynb, following the same
pattern as src/models.py, src/tuning.py, src/evaluation.py, and
src/experiment_tracking.py: the notebook calls these, displays results, and
does its Check verifications inline.
 
This module is the ONE thing Day 11 (dashboard) and Day 12 (API) should
import for making predictions -- it deliberately does NOT depend on mlflow,
src.data_loader, or any training code, only on the single portable bundle
file produced by 09_save_model.ipynb (models/ckd_pipeline.joblib). This means
the dashboard/API can start up without a working MLflow tracking store or
any of the training pipeline installed.
 
Typical usage:
 
    from src.inference import predict_sample, load_bundle
 
    bundle = load_bundle("models/ckd_pipeline.joblib")   # load once at startup
    result = predict_sample(raw_patient_dict, bundle=bundle)
"""
 
import joblib
import numpy as np
import pandas as pd
 
from src.features import encode_categoricals, add_domain_features
 
DEFAULT_BUNDLE_PATH = "models/ckd_pipeline.joblib"
 
 
# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------
 
def load_bundle(bundle_path: str = DEFAULT_BUNDLE_PATH) -> dict:
    """
    Load the full inference bundle (model + scaler + feature_order + metadata)
    saved by 09_save_model.ipynb. This is the only file the API/dashboard need
    -- no MLflow connection required at inference time.
    """
    bundle = joblib.load(bundle_path)
 
    required_keys = {"model", "scaler", "feature_order", "numeric_cols"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(
            f"Bundle at '{bundle_path}' is missing expected keys: {missing}. "
            "Was it saved by the current version of 09_save_model.ipynb?"
        )
 
    return bundle
 
 
# ---------------------------------------------------------------------------
# Preprocessing a single raw patient record
# ---------------------------------------------------------------------------
 
def _row_to_dataframe(raw_input: dict) -> pd.DataFrame:
    """Wrap a single patient's raw feature dict into a one-row DataFrame."""
    return pd.DataFrame([raw_input])
 
 
def preprocess_input(raw_input: dict, bundle: dict) -> pd.DataFrame:
    """
    Apply the exact same preprocessing used at training time to a single raw
    patient record: encode categoricals, add domain features (bun_creatinine
    ratio, anemia_ckd_flag), scale numeric columns with the FITTED scaler
    from the bundle (fit=False -- never re-fit at inference time), and
    reorder columns to match feature_order exactly.
 
    raw_input is expected to already be roughly "clean" (e.g. yes/no,
    normal/abnormal strings for categoricals, numeric values for labs) --
    the same shape as one row of data/processed/kidney_clean.csv before
    feature engineering, minus 'id' and 'classification'.
    """
    df = _row_to_dataframe(raw_input)
 
    df["classification"] = "notckd"  # placeholder, never used
    df = encode_categoricals(df)
    df = df.drop(columns=["classification"])
    df = add_domain_features(df)
 
    scaler = bundle["scaler"]
    numeric_cols = bundle["numeric_cols"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
 
    feature_order = bundle["feature_order"]
    missing_cols = set(feature_order) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Preprocessed input is missing expected columns: {missing_cols}. "
            "Check that raw_input includes every field the model was trained on."
        )
 
    # Reorder columns to match training-time order EXACTLY -- this is the
    # single most important line in this function. XGBoost/sklearn models
    # don't check column names at predict time by default; a silently
    # misordered DataFrame produces a confident, wrong prediction rather
    # than an error.
    df = df[feature_order]
 
    return df
 
 
# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
 
def predict_sample(
    raw_input: dict,
    bundle: dict = None,
    bundle_path: str = DEFAULT_BUNDLE_PATH,
) -> dict:
    """
    Run the full pipeline (preprocess + predict) on one raw patient record
    and return a plain-data result -- the exact shape Day 12's API endpoint
    and Day 16's LLM explainer will both consume.
 
    Pass a pre-loaded `bundle` (recommended for a running API/dashboard, so
    the model is loaded once at startup, not on every request) or leave it
    None to load fresh from bundle_path (convenient for notebooks/tests).
    """
    if bundle is None:
        bundle = load_bundle(bundle_path)
 
    processed = preprocess_input(raw_input, bundle)
    model = bundle["model"]
 
    prediction = int(model.predict(processed)[0])
    probability = (
        float(model.predict_proba(processed)[0][1])
        if hasattr(model, "predict_proba")
        else None
    )
 
    return {
        "prediction": prediction,
        "prediction_label": "ckd" if prediction == 1 else "notckd",
        "probability": probability,
        "model_name": bundle.get("model_name"),
        "mlflow_run_id": bundle.get("mlflow_run_id"),
    }
 
 
def predict_batch(
    raw_inputs: list,
    bundle: dict = None,
    bundle_path: str = DEFAULT_BUNDLE_PATH,
) -> list:
    """
    Same as predict_sample, but for a list of raw patient dicts -- loads the
    bundle once and reuses it across all rows, rather than once per row.
    Useful for Day 12's /predict-batch endpoint.
    """
    if bundle is None:
        bundle = load_bundle(bundle_path)
 
    return [predict_sample(row, bundle=bundle) for row in raw_inputs]
 
 
if __name__ == "__main__":
    # Smoke test: loads the ALREADY-SAVED bundle (does not retrain or
    # re-save anything) and confirms predict_sample() runs end-to-end
    # against a real row pulled from the processed test data.
    from src.data_loader import load_and_prepare
 
    bundle = load_bundle("models/ckd_pipeline.joblib")
    print("Loaded bundle. Model:", type(bundle["model"]).__name__)
    print("Feature order length:", len(bundle["feature_order"]))
 
    # NOTE: this smoke test pulls an ALREADY-PROCESSED test row (post
    # feature-engineering) purely to get realistic values for a manual
    # dict -- in real use, raw_input should be genuinely raw clinical
    # input, not a row that's already been through encode_categoricals/
    # scale_numeric. This is just for a quick standalone sanity check.
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(
        path="data/processed/kidney_features.csv", strategy="smote"
    )
 
    sample_processed_row = X_test.iloc[0]
    direct_pred = bundle["model"].predict(X_test.iloc[[0]])[0]
    direct_proba = bundle["model"].predict_proba(X_test.iloc[[0]])[0][1]
    print("Direct model call on an already-processed row:", int(direct_pred), float(direct_proba))
    print(
        "NOTE: predict_sample() expects RAW input and will re-run "
        "preprocessing -- see 09_save_model.ipynb Cell 8/9 for the correct "
        "end-to-end check using a genuinely raw patient record."
    )
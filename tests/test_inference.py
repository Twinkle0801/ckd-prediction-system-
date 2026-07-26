# tests/test_inference.py
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.inference import load_bundle, preprocess_input, predict_sample, predict_batch
from src.features import encode_categoricals, add_domain_features, NUMERIC_COLS

RAW_PATIENT = {
    "age": 48, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 121, "bu": 36, "sc": 1.2, "sod": 138, "pot": 4.4,
    "hemo": 15.4, "pcv": 44, "wc": 7800, "rc": 5.2,
    "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
}


@pytest.fixture
def fake_bundle():
    """
    Build a REAL (not mocked) bundle: fit a scaler + a logistic regression
    on synthetic data shaped like RAW_PATIENT after encode_categoricals +
    add_domain_features, so preprocess_input/predict_sample run through
    their actual code, not a stand-in.
    """
    rows = [RAW_PATIENT.copy() for _ in range(20)]
    df = pd.DataFrame(rows)
    df["classification"] = ["ckd", "notckd"] * 10
    encoded = encode_categoricals(df).drop(columns=["classification"])
    with_domain = add_domain_features(encoded)

    scaler = StandardScaler()
    scaler.fit(with_domain[NUMERIC_COLS])

    feature_order = list(with_domain.columns)
    X = with_domain.copy()
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    y = pd.Series([1, 0] * 10)
    model = LogisticRegression(max_iter=1000).fit(X[feature_order], y)

    return {
        "model": model,
        "scaler": scaler,
        "feature_order": feature_order,
        "numeric_cols": NUMERIC_COLS,
        "model_name": "logreg",
        "mlflow_run_id": "test-run-abc",
    }


def test_load_bundle_raises_on_missing_keys(tmp_path):
    incomplete_bundle = {"model": object(), "scaler": object()}  # missing feature_order, numeric_cols
    path = tmp_path / "bad_bundle.joblib"
    joblib.dump(incomplete_bundle, path)
    with pytest.raises(ValueError, match="missing expected keys"):
        load_bundle(str(path))


def test_load_bundle_succeeds_with_all_required_keys(tmp_path):
    complete_bundle = {
        "model": object(), "scaler": object(),
        "feature_order": ["a", "b"], "numeric_cols": ["a"],
    }
    path = tmp_path / "good_bundle.joblib"
    joblib.dump(complete_bundle, path)
    loaded = load_bundle(str(path))
    assert loaded["feature_order"] == ["a", "b"]


def test_preprocess_input_reorders_columns_to_feature_order(fake_bundle):
    processed = preprocess_input(RAW_PATIENT, fake_bundle)
    assert list(processed.columns) == fake_bundle["feature_order"]


def test_preprocess_input_scales_numeric_columns(fake_bundle):
    processed = preprocess_input(RAW_PATIENT, fake_bundle)
    # StandardScaler transform should change the raw value
    assert processed["hemo"].iloc[0] != RAW_PATIENT["hemo"]


def test_preprocess_input_raises_on_missing_required_field(fake_bundle):
    incomplete_patient = RAW_PATIENT.copy()
    del incomplete_patient["hemo"]
    with pytest.raises(KeyError):
        # add_domain_features needs 'hemo' -- must fail loudly, not silently
        # produce a wrong prediction from a missing column
        preprocess_input(incomplete_patient, fake_bundle)


def test_predict_sample_returns_expected_shape(fake_bundle):
    result = predict_sample(RAW_PATIENT, bundle=fake_bundle)
    assert set(result.keys()) == {
        "prediction", "prediction_label", "probability", "model_name", "mlflow_run_id"
    }
    assert result["prediction"] in (0, 1)
    assert result["model_name"] == "logreg"
    assert result["mlflow_run_id"] == "test-run-abc"


def test_predict_sample_label_matches_prediction_value(fake_bundle):
    result = predict_sample(RAW_PATIENT, bundle=fake_bundle)
    expected_label = "ckd" if result["prediction"] == 1 else "notckd"
    assert result["prediction_label"] == expected_label


def test_predict_batch_returns_one_result_per_input(fake_bundle):
    results = predict_batch([RAW_PATIENT, RAW_PATIENT, RAW_PATIENT], bundle=fake_bundle)
    assert len(results) == 3
    for r in results:
        assert "prediction" in r


def test_predict_batch_does_not_reload_bundle_when_one_is_passed(fake_bundle, monkeypatch):
    """predict_batch must reuse the passed-in bundle, not call load_bundle
    per row -- otherwise a real API with a slow joblib file would reload
    the model on every row of a large batch."""
    import src.inference as inference_module
    call_count = {"n": 0}
    original_load = inference_module.load_bundle

    def counting_load(*args, **kwargs):
        call_count["n"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(inference_module, "load_bundle", counting_load)
    predict_batch([RAW_PATIENT, RAW_PATIENT], bundle=fake_bundle)
    assert call_count["n"] == 0


def test_predict_sample_handles_model_without_predict_proba(fake_bundle):
    class NoProbaWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            return self.model.predict(X)

    fake_bundle_no_proba = dict(fake_bundle)
    fake_bundle_no_proba["model"] = NoProbaWrapper(fake_bundle["model"])
    result = predict_sample(RAW_PATIENT, bundle=fake_bundle_no_proba)
    assert result["probability"] is None
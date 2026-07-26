# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import api.main as main_module

VALID_PAYLOAD = {
    "age": 48, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 121, "bu": 36, "sc": 1.2, "sod": 138, "pot": 4.4,
    "hemo": 15.4, "pcv": 44, "wc": 7800, "rc": 5.2,
    "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
}

FAKE_PREDICT_RESULT = {
    "prediction": 1, "prediction_label": "ckd", "probability": 0.93,
    "model_name": "xgboost", "mlflow_run_id": "test-run-123",
}
FAKE_EXPLANATION = {
    "top_contributions": [
        {"feature": "hemo", "value": 15.4, "shap_contribution": -0.42},
        {"feature": "sc", "value": 1.2, "shap_contribution": 0.31},
    ]
}

@pytest.fixture
def client(monkeypatch):
    # Bypass the real lifespan — no real joblib file needed in tests
    main_module.bundle_store["bundle"] = {
        "model": object(), "model_name": "xgboost", "mlflow_run_id": "test-run-123",
        "feature_order": ["age", "bp", "hemo", "sc"],
    }
    monkeypatch.setattr(main_module, "predict_sample", lambda raw_input, bundle: FAKE_PREDICT_RESULT)
    monkeypatch.setattr(main_module, "preprocess_input", lambda raw_input, bundle: raw_input)
    monkeypatch.setattr(main_module, "explain_model", lambda model, processed: (None, None))
    monkeypatch.setattr(
        main_module, "explain_single_prediction",
        lambda model, explainer, shap_values, processed, idx: FAKE_EXPLANATION,
    )
    return TestClient(main_module.app)

def test_root_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "xgboost"
    assert body["n_features"] == 4

def test_predict_endpoint_valid_input_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["prediction_label"] == "ckd"
    assert len(body["top_factors"]) == 2
    assert body["top_factors"][0]["direction"] == "decreases_ckd_risk"  # shap_contribution < 0
    assert "decision support" in body["disclaimer"].lower()

def test_predict_endpoint_rejects_out_of_range_value(client):
    bad_payload = {**VALID_PAYLOAD, "sg": 5.0}  # sg must be <= 1.030 per schema
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_predict_endpoint_rejects_wrong_type(client):
    bad_payload = {**VALID_PAYLOAD, "age": "not_a_number"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_predict_endpoint_missing_field_returns_422(client):
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["hemo"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_predict_batch_endpoint_returns_matching_count(client):
    response = client.post("/predict-batch", json={"patients": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["results"]) == 3

def test_predict_batch_endpoint_empty_list(client):
    response = client.post("/predict-batch", json={"patients": []})
    assert response.status_code == 200
    assert response.json()["count"] == 0

def test_predict_endpoint_internal_error_returns_400(client, monkeypatch):
    def _boom(raw_input, bundle):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(main_module, "predict_sample", _boom)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 400
    assert "Prediction failed" in response.json()["detail"]

def test_predict_batch_endpoint_internal_error_returns_400(client, monkeypatch):
    def _boom(raw_input, bundle):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(main_module, "predict_sample", _boom)
    response = client.post("/predict-batch", json={"patients": [VALID_PAYLOAD]})
    assert response.status_code == 400
    assert "Batch prediction failed" in response.json()["detail"]
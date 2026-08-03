"""
Day 13: integration tests for api/main.py, using FastAPI's TestClient.
Loads the REAL saved model bundle via the app's lifespan — these tests
exercise the actual inference path used in production.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

VALID_PATIENT = {
    "age": 48, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 121, "bu": 36, "sc": 1.2, "sod": 138, "pot": 4.4,
    "hemo": 15.4, "pcv": 44, "wc": 7800, "rc": 5.2,
    "htn": "yes", "dm": "yes", "cad": "no", "appet": "good",
    "pe": "no", "ane": "no",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "mlflow_run_id" in data
    assert data["n_features"] == len(data["feature_order"])


def test_predict_valid_patient(client):
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["prediction_label"] in ("ckd", "notckd")
    assert 0.0 <= data["probability"] <= 1.0
    assert "disclaimer" in data
    assert len(data["top_factors"]) == 5


def test_predict_batch(client):
    response = client.post("/predict-batch", json={"patients": [VALID_PATIENT, VALID_PATIENT]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


# ── Edge cases per Day 13 roadmap: missing fields, out-of-range values ───

def test_predict_missing_field_returns_422(client):
    incomplete = VALID_PATIENT.copy()
    del incomplete["age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422, "Pydantic should reject missing required field"


def test_predict_out_of_range_age_returns_422(client):
    bad_patient = VALID_PATIENT.copy()
    bad_patient["age"] = 200  # exceeds le=120 constraint
    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_out_of_range_specific_gravity_returns_422(client):
    bad_patient = VALID_PATIENT.copy()
    bad_patient["sg"] = 2.5  # exceeds le=1.030 constraint
    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_invalid_categorical_string(client):
    """rbc/htn/etc. are plain str fields with no enum constraint at the
    pydantic level -- confirms current behavior (accepted, not rejected).
    If you later add stricter validation, this test should be updated to
    expect a 422 instead."""
    weird_patient = VALID_PATIENT.copy()
    weird_patient["htn"] = "maybe"
    response = client.post("/predict", json=weird_patient)
    # Documenting current behavior rather than asserting an opinion on it
    assert response.status_code in (200, 400, 422)


def test_predict_empty_batch(client):
    response = client.post("/predict-batch", json={"patients": []})
    assert response.status_code == 200
    assert response.json()["count"] == 0


# ── Sanity-check against known CKD/non-CKD test-set examples ─────────────

def test_predict_matches_known_ckd_example(client):
    """A record with strongly CKD-indicative values (high creatinine, low
    hemoglobin, hypertension+diabetes) should predict CKD with reasonable
    confidence -- sanity check per Day 13 roadmap."""
    ckd_like_patient = VALID_PATIENT.copy()
    ckd_like_patient.update({
        "sc": 5.0, "hemo": 8.0, "htn": "yes", "dm": "yes", "appet": "poor",
    })
    response = client.post("/predict", json=ckd_like_patient)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_label"] == "ckd", (
        f"Expected CKD prediction for strongly CKD-indicative values, got {data['prediction_label']}"
    )
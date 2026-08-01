"""
smoke_test_live.py

Standalone smoke test against the REAL deployed URLs -- not part of the
pytest suite in tests/, and deliberately so: this is the one thing that
can't be verified by TestClient, since it depends on actual DNS, network
path, and hosting-platform behavior (Render/Streamlit Cloud), not just
your code's logic.

Run with:
    python smoke_test_live.py
"""

import sys
import time

import requests

API_URL = "https://ckd-prediction-system-szyw.onrender.com"
DASHBOARD_URL = "https://g8r8qyh8sq4d4wvj22xqdw.streamlit.app"

VALID_PAYLOAD = {
    "age": 48, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 121, "bu": 36, "sc": 1.2, "sod": 138, "pot": 4.4,
    "hemo": 15.4, "pcv": 44, "wc": 7800, "rc": 5.2,
    "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
}


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def test_api_root():
    r = requests.get(f"{API_URL}/", timeout=90)  # generous timeout: free tier may be cold-starting
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    return check("API root (/) responds", ok, f"status={r.status_code}")


def test_api_model_info():
    r = requests.get(f"{API_URL}/model-info", timeout=60)
    ok = r.status_code == 200 and "model_name" in r.json()
    return check("API /model-info responds", ok, f"status={r.status_code}")


def test_api_predict():
    r = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, timeout=60)
    body = r.json() if r.status_code == 200 else {}
    ok = (
        r.status_code == 200
        and "prediction" in body
        and "probability" in body
        and "top_factors" in body
        and len(body["top_factors"]) > 0
    )
    return check("API /predict returns a real prediction", ok, f"status={r.status_code}")


def test_api_predict_batch():
    r = requests.post(
        f"{API_URL}/predict-batch",
        json={"patients": [VALID_PAYLOAD, VALID_PAYLOAD]},
        timeout=60,
    )
    body = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and body.get("count") == 2 and len(body.get("results", [])) == 2
    return check("API /predict-batch returns 2 results", ok, f"status={r.status_code}")


def test_api_rejects_invalid_input():
    bad_payload = {**VALID_PAYLOAD, "sg": 5.0}  # out of schema's allowed range
    r = requests.post(f"{API_URL}/predict", json=bad_payload, timeout=60)
    return check("API /predict rejects out-of-range input (422)", r.status_code == 422, f"status={r.status_code}")


def test_dashboard_loads():
    r = requests.get(DASHBOARD_URL, timeout=60)
    ok = r.status_code == 200 and "text/html" in r.headers.get("content-type", "")
    return check("Dashboard root URL loads", ok, f"status={r.status_code}")


if __name__ == "__main__":
    print(f"Running live smoke tests against:\n  API: {API_URL}\n  Dashboard: {DASHBOARD_URL}\n")
    start = time.time()

    results = [
        test_api_root(),
        test_api_model_info(),
        test_api_predict(),
        test_api_predict_batch(),
        test_api_rejects_invalid_input(),
        test_dashboard_loads(),
    ]

    elapsed = time.time() - start
    passed = sum(results)
    total = len(results)

    print(f"\n{passed}/{total} checks passed in {elapsed:.1f}s")

    if passed != total:
        sys.exit(1)
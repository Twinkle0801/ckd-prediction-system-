# tests/conftest.py
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def raw_ckd_row():
    """One synthetic raw row matching the CKD schema before cleaning."""
    return {
        "age": "48", "bp": "80", "sg": "1.020", "al": "1", "su": "0",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": "121", "bu": "36", "sc": "1.2", "sod": "137", "pot": "4.6",
        "hemo": "15.4", "pcv": "44", "wc": "7800", "rc": "5.2",
        "htn": "yes", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no", "classification": "ckd"
    }

@pytest.fixture
def raw_ckd_df(raw_ckd_row):
    return pd.DataFrame([raw_ckd_row] * 20)  # repeated rows so median/mode ops don't crash

@pytest.fixture
def cleaned_ckd_df(raw_ckd_df, tmp_path):
    from src.cleaning import clean_ckd_data
    path = tmp_path / "raw.csv"
    raw_ckd_df.to_csv(path, index=False)
    return clean_ckd_data(str(path))
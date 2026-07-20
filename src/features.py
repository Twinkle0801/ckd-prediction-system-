import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

BINARY_MAP = {"yes": 1, "no": 0, "good": 1, "poor": 0, "normal": 1, "abnormal": 0,
              "present": 1, "notpresent": 0, "ckd": 1, "notckd": 0}
CAT_COLS = ["htn", "dm", "cad", "appet", "pe", "ane", "rbc", "pc", "pcc", "ba", "classification"]
NUMERIC_COLS = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower().map(BINARY_MAP)

    # FIX: impute missing categoricals with the mode, excluding the target column
    impute_cols = [c for c in CAT_COLS if c != "classification"]
    for col in impute_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df

def scale_numeric(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    df = df.copy()
    if fit:
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df, scaler

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # FIX: compute the ratio on RAW lab values — this function must run
    # BEFORE scale_numeric now (see build_features below)
    df["bun_creatinine_ratio"] = df["bu"] / df["sc"].replace(0, np.nan)
    df["bun_creatinine_ratio"] = df["bun_creatinine_ratio"].fillna(df["bun_creatinine_ratio"].median())

    # FIX: anemia flag now uses raw hemoglobin (g/dL), not a scaled z-score threshold.
    # 12 g/dL is a common general low-hemoglobin cutoff — replace with your actual
    # clinical reference range if your data dictionary specifies a different one.
    df["anemia_ckd_flag"] = ((df["ane"] == 1) & (df["hemo"] < 12)).astype(int)
    return df

def build_features(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    df = encode_categoricals(df)
    df = add_domain_features(df)              # FIX: moved BEFORE scale_numeric
    df, scaler = scale_numeric(df, scaler=scaler, fit=fit)
    return df, scaler
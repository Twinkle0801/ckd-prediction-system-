
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
    df["bun_creatinine_ratio"] = df["bu"] / df["sc"]
    df["anemia_ckd_flag"] = ((df["ane"] == 1) & (df["hemo"] < -0.5)).astype(int)
    return df

def build_features(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    df = encode_categoricals(df)
    df, scaler = scale_numeric(df, scaler=scaler, fit=fit)
    df = add_domain_features(df)
    return df, scaler
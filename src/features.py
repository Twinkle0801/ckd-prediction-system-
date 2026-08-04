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

def build_features(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None, scale: bool = False):
    """
    Day 8 leakage fix: `scale` now defaults to False.

    Previously this function always scaled numeric columns, which meant any
    caller that ran it on the FULL dataset (e.g. the feature-engineering
    notebook, before any train/test split existed) silently baked a
    whole-dataset-fit StandardScaler into kidney_features.csv -- leaking
    test-set statistics into training.

    Scaling now happens ONLY in src/data_loader.py's load_and_prepare(),
    via scale_train_test(), fit on X_train only, AFTER the split. Pass
    scale=True explicitly only if you have a specific, deliberate reason
    to scale before that point -- and be sure fit/split has already happened
    for whatever df you're passing in.
    """
    df = encode_categoricals(df)
    df = add_domain_features(df)              # FIX: moved BEFORE scale_numeric
    if scale:
        df, scaler = scale_numeric(df, scaler=scaler, fit=fit)
    return df, scaler

def scale_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Fit StandardScaler on the training data only,
    then transform both training and testing data.
    """

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[NUMERIC_COLS] = scaler.fit_transform(
        X_train[NUMERIC_COLS]
    )

    X_test[NUMERIC_COLS] = scaler.transform(
        X_test[NUMERIC_COLS]
    )

    return X_train, X_test, scaler
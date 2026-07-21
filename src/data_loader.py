# src/data_loader.py
import shutil
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.features import scale_train_test

TARGET_COL = "classification"
ID_COL = "id"


# ---------------------------------------------------------------------------
# Day 1 — Data Collection
# ---------------------------------------------------------------------------

def fetch_ckd_dataset(output_path: str = "data/raw/kidney_disease.csv"):
    import kagglehub  # imported here, not at module top — only needed for this function
    path = kagglehub.dataset_download("mansoordaku/ckdisease")
    print("Downloaded to:", path)

    for f in os.listdir(path):
        if f.endswith(".csv"):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(os.path.join(path, f), output_path)
            print(f"Saved to {output_path}")
            return output_path


# ---------------------------------------------------------------------------
# Day 5 — Handle Class Imbalance & Train-Test Split
# ---------------------------------------------------------------------------

def load_features(path: str = "data/processed/kidney_features.csv") -> pd.DataFrame:
    """Load the engineered feature set produced by src/features.py (unscaled)."""
    df = pd.read_csv(path)
    return df


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL, id_col: str = ID_COL):
    """
    Separate features from target. Drops the row-identifier column, since it
    carries no clinical signal and would otherwise get synthesized by SMOTE
    or picked up as a spurious predictor by tree-based models.
    """
    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def check_missing(X: pd.DataFrame) -> pd.Series:
    """Return any columns with missing values (empty Series if none)."""
    null_counts = X.isnull().sum()
    return null_counts[null_counts > 0]


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split, preserving class ratio in both sets. Runs on RAW (unscaled) features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Balance the training set via SMOTE. Must only ever be called on
    training data — never on X_test/y_test, or it introduces leakage.
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def apply_undersampling(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Balance the training set via random undersampling. Same leakage rule
    as apply_smote — training data only.
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res


def verify_no_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> bool:
    """Confirm train and test indices never overlap. Returns True if clean."""
    overlap = set(X_train.index) & set(X_test.index)
    return len(overlap) == 0


def load_and_prepare(path: str = "data/processed/kidney_features.csv",
                      strategy: str = "smote",
                      test_size: float = 0.2,
                      random_state: int = 42):
    """
    End-to-end pipeline, in the CORRECT order:
      1. load raw (unscaled) features
      2. split into train/test
      3. scale — fit StandardScaler on TRAIN ONLY, transform test with that same scaler
      4. balance the TRAIN set only (SMOTE / undersample / none)

    strategy: "smote" | "undersample" | "none" (class weighting handled
    at model-training time instead, via class_weight="balanced")

    Returns: X_train_final, X_test_scaled, y_train_final, y_test, scaler
    """
    df = load_features(path)
    X, y = split_X_y(df)

    missing = check_missing(X)
    if len(missing) > 0:
        raise ValueError(f"Missing values found in columns: {missing.to_dict()}")

    # split BEFORE scaling
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, random_state)

    if not verify_no_leakage(X_train, X_test):
        raise RuntimeError("Train/test index overlap detected — split is invalid.")

    # FIX: scale AFTER splitting, fit on train only — this was previously missing
    # entirely, which is why load_and_prepare() only returned 4 values instead of 5
    # and every downstream model saw features on inconsistent scales.
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    # balance AFTER scaling, train only, so synthetic SMOTE rows are generated in
    # the same feature space the model will see at inference time
    if strategy == "smote":
        X_train_final, y_train_final = apply_smote(X_train_scaled, y_train, random_state)
    elif strategy == "undersample":
        X_train_final, y_train_final = apply_undersampling(X_train_scaled, y_train, random_state)
    elif strategy == "none":
        X_train_final, y_train_final = X_train_scaled, y_train
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X_train_final, X_test_scaled, y_train_final, y_test, scaler


if __name__ == "__main__":
    fetch_ckd_dataset()

    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy="smote")
    print("Train shape:", X_train.shape, "| class counts:", y_train.value_counts().to_dict())
    print("Test shape :", X_test.shape, "| class counts:", y_test.value_counts().to_dict())
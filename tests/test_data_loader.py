# tests/test_data_loader.py
import pandas as pd
import numpy as np
import pytest
from src.data_loader import (
    split_X_y, check_missing, stratified_split, apply_smote,
    apply_undersampling, verify_no_leakage, load_and_prepare, TARGET_COL, ID_COL,
)
from src.features import NUMERIC_COLS  # scale_train_test needs ALL of these present


@pytest.fixture
def sample_features_df():
    """
    Synthetic dataset that mirrors the real kidney_features.csv shape:
    must include every column in NUMERIC_COLS (scale_train_test selects
    them by name), plus id and the target column.
    """
    n = 60
    rng = np.random.default_rng(42)
    data = {col: rng.random(n) * 50 + 1 for col in NUMERIC_COLS}
    data["id"] = range(n)
    data[TARGET_COL] = [0, 1] * (n // 2)
    return pd.DataFrame(data)


def test_split_X_y_drops_target_and_id(sample_features_df):
    X, y = split_X_y(sample_features_df)
    assert TARGET_COL not in X.columns
    assert ID_COL not in X.columns
    assert len(y) == len(sample_features_df)


def test_split_X_y_handles_missing_id_column(sample_features_df):
    df_no_id = sample_features_df.drop(columns=[ID_COL])
    X, y = split_X_y(df_no_id)  # must not raise even though id_col isn't present
    assert TARGET_COL not in X.columns


def test_check_missing_flags_only_columns_with_nans(sample_features_df):
    df = sample_features_df.copy()
    df.loc[0, "hemo"] = np.nan
    missing = check_missing(df.drop(columns=[TARGET_COL, ID_COL]))
    assert "hemo" in missing.index
    assert "age" not in missing.index


def test_stratified_split_preserves_class_ratio(sample_features_df):
    X, y = split_X_y(sample_features_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    assert abs(y_train.mean() - y_test.mean()) < 0.15  # loose bound for a small synthetic sample


def test_verify_no_leakage_detects_clean_split(sample_features_df):
    X, y = split_X_y(sample_features_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    assert verify_no_leakage(X_train, X_test) is True


def test_verify_no_leakage_catches_overlap(sample_features_df):
    X, y = split_X_y(sample_features_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    X_test_with_dupe = pd.concat([X_test, X_train.iloc[[0]]])  # inject a leaked row
    assert verify_no_leakage(X_train, X_test_with_dupe) is False


def test_apply_smote_balances_classes():
    """
    SMOTE's default k_neighbors=5 requires MORE than 5 samples in the
    minority class (n_samples_fit must exceed k_neighbors), so the
    minority class here uses 10 samples, not 5.
    """
    X_train = pd.DataFrame(np.random.rand(30, 3), columns=list("abc"))
    y_train = pd.Series([1] * 20 + [0] * 10)  # imbalanced but SMOTE-safe
    X_res, y_res = apply_smote(X_train, y_train)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]


def test_apply_undersampling_balances_classes():
    X_train = pd.DataFrame(np.random.rand(30, 3), columns=list("abc"))
    y_train = pd.Series([1] * 25 + [0] * 5)
    X_res, y_res = apply_undersampling(X_train, y_train)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(y_res) == 10  # shrinks to 2x minority class


def test_load_and_prepare_raises_on_missing_values(tmp_path, sample_features_df):
    df = sample_features_df.copy()
    df.loc[0, "hemo"] = np.nan
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing values"):
        load_and_prepare(path=str(path))


def test_load_and_prepare_returns_five_values_and_no_leakage(tmp_path, sample_features_df):
    path = tmp_path / "features.csv"
    sample_features_df.to_csv(path, index=False)
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(path=str(path), strategy="smote")
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert scaler is not None  # confirms the FIX: scaling now actually happens


def test_load_and_prepare_unknown_strategy_raises(tmp_path, sample_features_df):
    path = tmp_path / "features.csv"
    sample_features_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown strategy"):
        load_and_prepare(path=str(path), strategy="bogus")
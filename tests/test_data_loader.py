"""
Day 13: unit + integration tests for src/data_loader.py
"""
import pandas as pd
import pytest
from src.data_loader import (
    load_features,
    split_X_y,
    check_missing,
    stratified_split,
    apply_smote,
    apply_undersampling,
    verify_no_leakage,
    load_and_prepare,
    TARGET_COL,
    ID_COL,
)


@pytest.fixture
def raw_df():
    return load_features()


def test_load_features_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) > 0


def test_split_X_y_drops_target_and_id(raw_df):
    X, y = split_X_y(raw_df)
    assert TARGET_COL not in X.columns
    assert ID_COL not in X.columns
    assert len(X) == len(y)


def test_check_missing_on_clean_data(raw_df):
    X, _ = split_X_y(raw_df)
    missing = check_missing(X)
    assert missing.empty, f"Unexpected missing values: {missing.to_dict()}"


def test_stratified_split_preserves_class_ratio(raw_df):
    X, y = split_X_y(raw_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=42)

    full_ratio = y.value_counts(normalize=True)
    train_ratio = y_train.value_counts(normalize=True)
    test_ratio = y_test.value_counts(normalize=True)

    for cls in full_ratio.index:
        assert abs(train_ratio[cls] - full_ratio[cls]) < 0.05
        assert abs(test_ratio[cls] - full_ratio[cls]) < 0.05


def test_verify_no_leakage_detects_clean_split(raw_df):
    X, y = split_X_y(raw_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    assert verify_no_leakage(X_train, X_test) is True


def test_verify_no_leakage_catches_overlap():
    df = pd.DataFrame({"a": [1, 2, 3]})
    overlapping = df.copy()
    assert verify_no_leakage(df, overlapping) is False


def test_apply_smote_balances_classes(raw_df):
    X, y = split_X_y(raw_df)
    X_train, _, y_train, _ = stratified_split(X, y)
    X_res, y_res = apply_smote(X_train, y_train)
    counts = y_res.value_counts()
    assert counts.min() == counts.max(), "SMOTE should fully balance classes"


def test_apply_undersampling_balances_classes(raw_df):
    X, y = split_X_y(raw_df)
    X_train, _, y_train, _ = stratified_split(X, y)
    X_res, y_res = apply_undersampling(X_train, y_train)
    counts = y_res.value_counts()
    assert counts.min() == counts.max()


# ── Integration tests: the real end-to-end pipeline ──────────────────────

@pytest.mark.parametrize("strategy", ["smote", "undersample", "none"])
def test_load_and_prepare_all_strategies(strategy):
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy=strategy)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert scaler is not None


def test_load_and_prepare_no_leakage_between_splits():
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy="none")
    # X_train may have new indices post-SMOTE for other strategies, but with
    # strategy="none" indices are preserved from the original split
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_load_and_prepare_scaling_applied():
    """Day 8 leakage-fix regression test: confirm scaling happens INSIDE
    load_and_prepare (train-only), not upstream in kidney_features.csv."""
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(strategy="none")
    # scaled numeric columns should have near-zero mean on the TRAINING set
    # (scaler was fit on X_train, so X_train's numeric cols center near 0)
    from src.features import NUMERIC_COLS
    means = X_train[NUMERIC_COLS].mean().abs()
    assert (means < 0.5).all(), "Training data does not look scaled — possible regression of Day 8 fix"


def test_load_and_prepare_invalid_strategy_raises():
    with pytest.raises(ValueError):
        load_and_prepare(strategy="not_a_real_strategy")


def test_smote_never_touches_test_set():
    """Regression test for the exact leakage pattern this project already
    found and fixed once (Day 8) — test set size must stay fixed regardless
    of the balancing strategy applied to training data only."""
    _, X_test_smote, _, y_test_smote, _ = load_and_prepare(strategy="smote")
    _, X_test_none, _, y_test_none, _ = load_and_prepare(strategy="none")
    assert len(X_test_smote) == len(X_test_none)
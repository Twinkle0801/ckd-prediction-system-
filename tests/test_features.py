# tests/test_features.py
import pandas as pd
import numpy as np
from src.features import (
    encode_categoricals, scale_numeric, add_domain_features,
    build_features, BINARY_MAP, NUMERIC_COLS
)

def test_encode_categoricals_maps_binary_values(cleaned_ckd_df):
    encoded = encode_categoricals(cleaned_ckd_df)
    assert set(encoded["htn"].unique()).issubset({0, 1})
    assert set(encoded["classification"].unique()).issubset({0, 1})

def test_encode_categoricals_no_nan_in_non_target_cols(cleaned_ckd_df):
    encoded = encode_categoricals(cleaned_ckd_df)
    non_target = [c for c in encoded.columns if c != "classification"]
    assert encoded[non_target].isna().sum().sum() == 0

def test_domain_features_computed_before_scaling(cleaned_ckd_df):
    """
    Regression test for the ordering bug the FIX comments call out:
    add_domain_features must run on RAW (unscaled) bu/sc/hemo values.
    """
    encoded = encode_categoricals(cleaned_ckd_df)
    with_domain = add_domain_features(encoded)
    expected_ratio = (encoded["bu"] / encoded["sc"].replace(0, np.nan)).fillna(
        (encoded["bu"] / encoded["sc"].replace(0, np.nan)).median()
    )
    pd.testing.assert_series_equal(
        with_domain["bun_creatinine_ratio"].reset_index(drop=True),
        expected_ratio.reset_index(drop=True),
        check_names=False
    )

def test_anemia_flag_uses_raw_hemoglobin_not_scaled(cleaned_ckd_df):
    encoded = encode_categoricals(cleaned_ckd_df)
    with_domain = add_domain_features(encoded)
    manual_flag = ((encoded["ane"] == 1) & (encoded["hemo"] < 12)).astype(int)
    pd.testing.assert_series_equal(
        with_domain["anemia_ckd_flag"].reset_index(drop=True),
        manual_flag.reset_index(drop=True),
        check_names=False
    )

def test_build_features_runs_domain_before_scale(cleaned_ckd_df):
    """Full pipeline order check — this is the test that would have caught
    the original bug if domain features were computed AFTER scaling."""
    built, scaler = build_features(cleaned_ckd_df, fit=True, scale=True)
    # after scaling, numeric cols should have ~zero mean (StandardScaler property)
    assert abs(built[NUMERIC_COLS].mean().mean()) < 1e-6
    # domain feature should still exist and be finite (not NaN from a bad scale-then-divide)
    assert built["bun_creatinine_ratio"].notna().all()

def test_scale_train_test_fits_only_on_train(cleaned_ckd_df):
    from src.features import scale_train_test
    train_df = cleaned_ckd_df.iloc[:15].copy()
    test_df = cleaned_ckd_df.iloc[15:].copy()
    X_train, X_test, scaler = scale_train_test(train_df, test_df)
    # scaler's learned mean_ should reflect only train_df's numeric columns
    manual_mean = train_df[NUMERIC_COLS].mean().values
    np.testing.assert_allclose(scaler.mean_, manual_mean, rtol=1e-5)

def test_scale_numeric_transform_only_path(cleaned_ckd_df):
    from src.features import encode_categoricals, scale_numeric
    encoded = encode_categoricals(cleaned_ckd_df)
    fitted_df, scaler = scale_numeric(encoded, fit=True)
    transformed_df, same_scaler = scale_numeric(encoded, scaler=scaler, fit=False)
    assert same_scaler is scaler  # fit=False must reuse the passed-in scaler, not create a new one
# tests/test_cleaning.py
import pandas as pd
from src.cleaning import clean_ckd_data, NUMERIC_COLS, CATEGORICAL_COLS

def test_clean_ckd_data_no_missing_values(cleaned_ckd_df):
    assert cleaned_ckd_df[NUMERIC_COLS + CATEGORICAL_COLS].isna().sum().sum() == 0

def test_numeric_cols_are_numeric_dtype(cleaned_ckd_df):
    for col in NUMERIC_COLS:
        assert pd.api.types.is_numeric_dtype(cleaned_ckd_df[col])

def test_categorical_text_is_stripped(cleaned_ckd_df):
    # no leading/trailing whitespace or stray tabs should remain
    for col in CATEGORICAL_COLS:
        assert cleaned_ckd_df[col].astype(str).str.contains(r"^\s|\s$|\t").sum() == 0

def test_question_marks_are_treated_as_missing(tmp_path):
    from src.cleaning import clean_ckd_data
    df = pd.DataFrame({**{c: ["1"] * 5 for c in [
        "age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]},
        **{c: ["normal"] * 4 + ["?"] for c in [
        "rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]}})
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    cleaned = clean_ckd_data(str(path))
    assert cleaned["rbc"].iloc[-1] == cleaned["rbc"].mode()[0]  # '?' got imputed, not left as-is
import pandas as pd

NUMERIC_COLS = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot',
                 'hemo','pcv','wc','rc']
CATEGORICAL_COLS = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe',
                     'ane','classification']

def clean_ckd_data(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Step 2: standardize categorical text
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string").str.strip().str.replace('\t', '', regex=False)
        df[col] = df[col].replace('?', pd.NA)
    # Step 3: fix numeric dtypes
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 4: impute
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
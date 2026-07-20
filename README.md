## Data Dictionary

Source: [Kaggle - CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease) (derived from UCI ML Repository)
Shape: 400 rows × 26 columns (24 features + `id` + target `classification`)

| Column | Full Name | Unit | Normal Range | Type |
|--------|-----------|------|---------------|------|
| age | Age | years | - | numeric |
| bp | Blood Pressure | mm/Hg | 90–120 (systolic, approx) | numeric |
| sg | Specific Gravity | - | 1.005–1.030 | numeric (ordinal-like) |
| al | Albumin | - (grade 0–5) | 0 (0 = normal) | numeric |
| su | Sugar | - (grade 0–5) | 0 (0 = normal) | numeric |
| rbc | Red Blood Cells | - | normal / abnormal | categorical |
| pc | Pus Cell | - | normal / abnormal | categorical |
| pcc | Pus Cell Clumps | - | present / notpresent | categorical |
| ba | Bacteria | - | present / notpresent | categorical |
| bgr | Blood Glucose Random | mg/dL | 70–140 | numeric |
| bu | Blood Urea | mg/dL | 7–20 | numeric |
| sc | Serum Creatinine | mg/dL | 0.6–1.3 | numeric |
| sod | Sodium | mEq/L | 135–145 | numeric |
| pot | Potassium | mEq/L | 3.5–5.0 | numeric |
| hemo | Hemoglobin | g/dL | 13.5–17.5 (M), 12–15.5 (F) | numeric |
| pcv | Packed Cell Volume | % | 38–50 (M), 34–44 (F) | numeric |
| wc | White Blood Cell Count | cells/cumm | 4,500–11,000 | numeric |
| rc | Red Blood Cell Count | millions/cumm | 4.5–5.9 (M), 4.0–5.2 (F) | numeric |
| htn | Hypertension | - | yes / no | categorical |
| dm | Diabetes Mellitus | - | yes / no | categorical |
| cad | Coronary Artery Disease | - | yes / no | categorical |
| appet | Appetite | - | good / poor | categorical |
| pe | Pedal Edema | - | yes / no | categorical |
| ane | Anemia | - | yes / no | categorical |
| classification | Target label | - | ckd / notckd | categorical (target) |

**Target encoding**: `ckd` → 1, `notckd` → 0 (to be applied in Phase 4 - Feature Engineering)

**Known data issues (found during Day 1 inspection)**:
- Class balance: 248 `ckd` (62%) vs 150 `notckd` (37.5%) + 2 rows with `ckd\t` (tab artifact, needs stripping in Phase 2)
- Missing values present as blank/NaN cells across most numeric lab columns (not literal `?` marks, unlike the raw UCI `.arff` version)
- `id` column is a row index, not a clinical feature — should be dropped before modeling

## How to Reproduce

The cleaned dataset (`data/processed/kidney_clean.csv`) is **not committed to git** — it's a regenerable build artifact, not source data. To regenerate it locally:

```python
from src.cleaning import clean_ckd_data

df_clean = clean_ckd_data("data/raw/kidney_disease.csv")
df_clean.to_csv("data/processed/kidney_clean.csv", index=False)
```

This runs the full cleaning pipeline (categorical standardization, dtype fixes, missing-value imputation) defined in `src/cleaning.py` and writes the output to `data/processed/`.

### Feature Selection: Correlation Pruning

Checked pairwise correlations among numeric features using a 0.9 absolute-correlation threshold 
(standard cutoff for flagging near-duplicate/redundant predictors).

**Result:** No feature pairs exceeded 0.9, so no columns were dropped at this stage.

**Notable pair reviewed manually** (expected to be correlated given known physiology):
- `hemo` (hemoglobin) vs `pcv` (packed cell volume): corr = **0.847** — kept both, below threshold. 
  These measure overlapping but not identical physiology (hemoglobin concentration vs. red blood 
  cell volume fraction), so retaining both is defensible at this correlation level.

**Why 0.9 and not lower (e.g. 0.7–0.8):** a stricter threshold risks discarding clinically 
meaningful features that are correlated but not redundant — in a recall-first CKD model, 
losing a weak-but-real signal is a bigger risk than keeping two moderately correlated labs.
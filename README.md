# AI-Powered Kidney Disease (CKD) Prediction System

An end-to-end machine learning system for predicting Chronic Kidney Disease (CKD) risk from clinical lab values, with a recall-first evaluation strategy, SHAP-based explainability, a FastAPI REST layer, and a Streamlit clinical decision-support dashboard.

**This is a decision-support tool only — it is NOT a medical diagnosis and should never replace a qualified healthcare professional.**

## Live Demo

- **Dashboard:** https://g8r8qyh8sq4d4wvj22xqdw.streamlit.app
- **API (Swagger docs):** https://ckd-prediction-system-szyw.onrender.com/docs

> Note: the API runs on Render's free tier, which spins down after periods of inactivity — the first request after idle time may take 30-60 seconds to respond while it wakes up.

## Project Overview

Most "kidney disease prediction" projects stop at training a classifier and reporting accuracy. This project goes further by treating a missed CKD case as far costlier than a false alarm, and building the supporting infrastructure a real clinical decision-support tool would need:

- **Recall-first evaluation** — models are compared and tuned to prioritize catching true CKD cases, not raw accuracy
- **Explainability** — every prediction comes with SHAP-based, per-patient feature contributions, not just a bare label
- **Reproducible experimentation** — every model run is tracked with MLflow (params, metrics, artifacts)
- **Production-shaped serving** — a versioned inference pipeline (`src/inference.py`) backs both a REST API and an interactive dashboard, so predictions are computed identically regardless of entry point
- **Tested** — unit and integration tests across the full pipeline (cleaning, feature engineering, tuning, models, inference, MLflow tracking, and the API), not just a notebook that "ran once"
- **Containerized & deployed** — both services run in Docker locally and are deployed publicly (Render + Streamlit Community Cloud)

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python (pandas, numpy) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | imbalanced-learn (SMOTE / undersampling) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| API | FastAPI + Uvicorn |
| Testing | pytest, pytest-cov |
| Containerization | Docker, Docker Compose |
| Deployment | Render (API), Streamlit Community Cloud (dashboard) |
| Version Control | Git & GitHub |

## Project Structure

```
ckd-prediction-system/
├── data/
│   ├── raw/                  # original dataset (not committed)
│   └── processed/            # cleaned/engineered data (regenerable, not committed)
├── notebooks/                # exploratory + build notebooks, one per phase
├── src/
│   ├── cleaning.py           # Day 2 — data cleaning pipeline
│   ├── features.py           # Day 4 — feature engineering
│   ├── data_loader.py        # Day 5 — split, scale, resample
│   ├── models.py              # Day 6 — model registry, train/evaluate
│   ├── tuning.py              # Day 7 — hyperparameter search
│   ├── evaluation.py          # Day 8 — leakage checks, SHAP, model card
│   ├── experiment_tracking.py # Day 9 — MLflow logging/registry
│   └── inference.py           # Day 10 — portable inference bundle
├── api/
│   ├── main.py                # Day 12 — FastAPI app
│   └── schemas.py             # pydantic request/response models
├── tests/                     # Day 13 — pytest suite (73 tests)
├── models/
│   └── ckd_pipeline.joblib    # serialized inference bundle (model + scaler + feature order)
├── assets/                    # logo, custom CSS for the dashboard
├── app.py                     # Day 11 — Streamlit dashboard
├── Dockerfile.api              # Day 14 — API container
├── Dockerfile.dashboard        # Day 14 — dashboard container
├── docker-compose.yml          # Day 14 — run both services together
├── requirements.txt            # full dev/training dependency set
├── requirements-docker.txt     # trimmed dependency set used by both Dockerfiles
├── MODEL_CARD.md
└── README.md
```

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

**Target encoding**: `ckd` → 1, `notckd` → 0 (applied in Phase 4 - Feature Engineering)

**Known data issues (found during Day 1 inspection)**:
- Class balance: 248 `ckd` (62%) vs 150 `notckd` (37.5%) + 2 rows with `ckd\t` (tab artifact, stripped in Phase 2)
- Missing values present as blank/NaN cells across most numeric lab columns (not literal `?` marks, unlike the raw UCI `.arff` version)
- `id` column is a row index, not a clinical feature — dropped before modeling

## Feature Selection: Correlation Pruning

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

## How to Reproduce

### 1. Clone and set up the environment

```bash
git clone https://github.com/Twinkle0801/ckd-prediction-system-.git
cd ckd-prediction-system-
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

### 2. Regenerate the cleaned dataset

The cleaned dataset (`data/processed/kidney_clean.csv`) is **not committed to git** — it's a regenerable build artifact, not source data. To regenerate it locally:

```python
from src.cleaning import clean_ckd_data

df_clean = clean_ckd_data("data/raw/kidney_disease.csv")
df_clean.to_csv("data/processed/kidney_clean.csv", index=False)
```

This runs the full cleaning pipeline (categorical standardization, dtype fixes, missing-value imputation) defined in `src/cleaning.py` and writes the output to `data/processed/`.

### 3. Run the dashboard locally

```bash
streamlit run app.py
```

### 4. Run the API locally

```bash
uvicorn api.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive Swagger documentation.

### 5. Run the test suite

```bash
pytest --cov=src --cov=api --cov-report=term-missing
```

73 tests across data cleaning, feature engineering, hyperparameter tuning, model training/evaluation, data loading, inference, model explainability, MLflow experiment tracking, and the REST API. ~85% coverage across `src/` and `api/`.

### 6. Run both services in Docker

```bash
docker compose build
docker compose up
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Model

- **Final model:** XGBoost, selected on a recall-first basis after comparing Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, and LightGBM, both untuned and tuned via `RandomizedSearchCV`.
- **Class imbalance handling:** SMOTE applied to the training set only, after the train/test split and after scaling — never on the held-out test set.
- **Explainability:** SHAP `TreeExplainer` provides both global feature importance and per-patient local explanations, surfaced in the dashboard and via the API's `top_factors` field.
- **Full details:** see [`MODEL_CARD.md`](./MODEL_CARD.md) for metrics, known limitations, and intended use.

## Testing & Quality

- 73 automated tests (`pytest`), covering every `src/` module and the FastAPI layer
- Regression tests explicitly locking in real bugs found and fixed during development (e.g. a model-registry mutation bug, a feature-scaling order bug, and MLflow's duplicate-run behavior)
- ~85% test coverage across `src/` and `api/`

## Deployment

- **API** — containerized with `Dockerfile.api`, image pushed to GitHub Container Registry, deployed on Render (Docker runtime, free tier)
- **Dashboard** — containerized with `Dockerfile.dashboard` for local Docker Compose use; deployed separately on Streamlit Community Cloud (which builds directly from `requirements.txt`, not the Dockerfile)
- Both services can also be run together locally via `docker-compose.yml`

## Disclaimer

This application is intended for **educational and clinical decision-support purposes only**. It estimates CKD risk using a trained machine learning model and **is not a medical diagnosis**. Always consult a qualified healthcare professional for medical decisions.
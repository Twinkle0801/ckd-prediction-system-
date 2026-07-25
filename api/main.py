# api/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.inference import load_bundle, predict_sample, predict_batch, preprocess_input
from src.evaluation import explain_model, explain_single_prediction

from api.schemas import (
    PatientInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    ContributingFactor,
)

bundle_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pipeline ONCE at startup, not per-request
    bundle_store["bundle"] = load_bundle("models/ckd_pipeline.joblib")
    print("Model bundle loaded:", bundle_store["bundle"]["model_name"])
    yield
    bundle_store.clear()


app = FastAPI(
    title="CKD Prediction API",
    description="Decision-support API for Chronic Kidney Disease risk prediction. "
                 "Not a substitute for professional medical diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_prediction_response(raw_input: dict) -> PredictionResponse:
    bundle = bundle_store["bundle"]
    result = predict_sample(raw_input, bundle=bundle)

    processed = preprocess_input(raw_input, bundle)
    explainer, shap_values = explain_model(bundle["model"], processed)
    explanation = explain_single_prediction(bundle["model"], explainer, shap_values, processed, 0)

    top_factors = [
        ContributingFactor(
            feature=item["feature"],
            value=raw_input.get(item["feature"], item["value"]),
            shap_contribution=item["shap_contribution"],
            direction="increases_ckd_risk" if item["shap_contribution"] > 0 else "decreases_ckd_risk",
        )
        for item in explanation["top_contributions"][:5]
    ]

    return PredictionResponse(
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        probability=result["probability"],
        model_name=result["model_name"],
        mlflow_run_id=result["mlflow_run_id"],
        top_factors=top_factors,
    )


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "CKD Prediction API is running. See /docs for usage."}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["info"])
def model_info():
    bundle = bundle_store["bundle"]
    return ModelInfoResponse(
        model_name=bundle["model_name"],
        mlflow_run_id=bundle["mlflow_run_id"],
        n_features=len(bundle["feature_order"]),
        feature_order=bundle["feature_order"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(patient: PatientInput):
    try:
        return _build_prediction_response(patient.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch_endpoint(request: BatchPredictionRequest):
    try:
        results = [
            _build_prediction_response(patient.model_dump())
            for patient in request.patients
        ]
        return BatchPredictionResponse(results=results, count=len(results))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")
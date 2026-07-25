# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import List, Union


class PatientInput(BaseModel):
    """Raw clinical input -- one patient record, 24 fields, matching the
    training data's raw columns before feature engineering."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    bp: float = Field(..., ge=40, le=250, description="Blood pressure (mm/Hg)")
    sg: float = Field(..., ge=1.000, le=1.030, description="Specific gravity")
    al: int = Field(..., ge=0, le=5, description="Albumin (0=none, higher=worse)")
    su: int = Field(..., ge=0, le=5, description="Sugar (0=none, higher=worse)")
    rbc: str = Field(..., description="Red blood cells: 'normal' or 'abnormal'")
    pc: str = Field(..., description="Pus cell: 'normal' or 'abnormal'")
    pcc: str = Field(..., description="Pus cell clumps: 'present' or 'notpresent'")
    ba: str = Field(..., description="Bacteria: 'present' or 'notpresent'")
    bgr: float = Field(..., ge=20, le=500, description="Blood glucose random (mg/dl)")
    bu: float = Field(..., ge=1, le=400, description="Blood urea (mg/dl)")
    sc: float = Field(..., ge=0.1, le=30, description="Serum creatinine (mg/dl)")
    sod: float = Field(..., ge=100, le=170, description="Sodium (mEq/L)")
    pot: float = Field(..., ge=1, le=30, description="Potassium (mEq/L)")
    hemo: float = Field(..., ge=3, le=20, description="Hemoglobin (g/dl)")
    pcv: float = Field(..., ge=10, le=60, description="Packed cell volume (%)")
    wc: float = Field(..., ge=2000, le=25000, description="WBC count (cells/cmm)")
    rc: float = Field(..., ge=2, le=8, description="RBC count (millions/cmm)")
    htn: str = Field(..., description="Hypertension: 'yes' or 'no'")
    dm: str = Field(..., description="Diabetes mellitus: 'yes' or 'no'")
    cad: str = Field(..., description="Coronary artery disease: 'yes' or 'no'")
    appet: str = Field(..., description="Appetite: 'good' or 'poor'")
    pe: str = Field(..., description="Pedal edema: 'yes' or 'no'")
    ane: str = Field(..., description="Anemia: 'yes' or 'no'")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 48, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
                "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
                "bgr": 121, "bu": 36, "sc": 1.2, "sod": 138, "pot": 4.4,
                "hemo": 15.4, "pcv": 44, "wc": 7800, "rc": 5.2,
                "htn": "yes", "dm": "yes", "cad": "no", "appet": "good",
                "pe": "no", "ane": "no",
            }
        }


class ContributingFactor(BaseModel):
    feature: str
    value: Union[float, str]   # numeric labs OR raw categorical strings like "yes"/"normal"
    shap_contribution: float
    direction: str


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    model_name: str
    mlflow_run_id: str
    top_factors: List[ContributingFactor]
    disclaimer: str = (
        "This is decision support only and not a medical diagnosis. "
        "Please consult a healthcare professional."
    )


class BatchPredictionRequest(BaseModel):
    patients: List[PatientInput]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    count: int


class ModelInfoResponse(BaseModel):
    model_name: str
    mlflow_run_id: str
    n_features: int
    feature_order: List[str]
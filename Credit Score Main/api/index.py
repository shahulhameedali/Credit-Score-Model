from pathlib import Path
from typing import Dict
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    BASE_DIR
    / "Credit Score"
    / "model_outputs"
    / "random_forest_credit_model.joblib"
)

app = FastAPI(title="Credit Risk Analyzer")


class CreditData(BaseModel):
    age: int = Field(alias="AGE", ge=18, le=100)
    annual_income: float = Field(alias="ANNUAL_INCOME", ge=0)
    employment_type: int = Field(alias="EMPLOYMENT_TYPE")
    cibil_score: int = Field(alias="CIBIL_SCORE", ge=300, le=900)
    existing_loan_amount: float = Field(alias="EXISTING_LOAN_AMOUNT", ge=0)
    existing_emi: float = Field(alias="EXISTING_EMI", ge=0)
    requested_loan_amount: float = Field(alias="REQUESTED_LOAN_AMOUNT", ge=0)
    loan_tenure_months: int = Field(alias="LOAN_TENURE_MONTHS", ge=1)
    dependents: int = Field(alias="DEPENDENTS", ge=0)
    education: int = Field(alias="EDUCATION")
    marital_status: int = Field(alias="MARRIAGE")

    model_config = {"populate_by_name": True}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = BASE_DIR / "public" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/predict")
def predict(data: CreditData) -> Dict[str, object]:
    try:
        model = load_model()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    values = data.model_dump(by_alias=True)

    # Feature engineering for SBI model
    values["EMI_TO_INCOME_RATIO"] = values["EXISTING_EMI"] / (values["ANNUAL_INCOME"] / 12)
    values["LOAN_TO_INCOME_RATIO"] = values["REQUESTED_LOAN_AMOUNT"] / values["ANNUAL_INCOME"]
    values["TOTAL_DEBT_BURDEN"] = values["EXISTING_LOAN_AMOUNT"] + values["REQUESTED_LOAN_AMOUNT"]

    # CIBIL risk tier
    cibil = values["CIBIL_SCORE"]
    if cibil < 550:
        values["CIBIL_RISK_TIER"] = 1
    elif cibil < 700:
        values["CIBIL_RISK_TIER"] = 2
    else:
        values["CIBIL_RISK_TIER"] = 3

    features = list(model.feature_names_in_)
    missing_features = [feature for feature in features if feature not in values]
    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Model features missing: {', '.join(missing_features)}",
        )

    input_df = pd.DataFrame([[values[feature] for feature in features]], columns=features)
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    default_probability = (
        float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    )
    non_default_probability = (
        float(probabilities[classes.index(0)]) if 0 in classes else 1.0 - default_probability
    )

    return {
        "prediction": prediction,
        "label": "Default" if prediction == 1 else "No Default",
        "default_probability": default_probability,
        "non_default_probability": non_default_probability,
        "risk_level": (
            "High" if default_probability >= 0.70
            else "Medium" if default_probability >= 0.40
            else "Low"
        ),
    }

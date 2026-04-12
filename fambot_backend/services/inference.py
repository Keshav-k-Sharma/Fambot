from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from fambot_backend.schemas import OnboardingIn

_ROOT = Path(__file__).resolve().parents[2]
_DATA_CSV = _ROOT / "sources" / "diabetes.csv"
_DEFAULT_MODEL_PATH = _ROOT / "diabetes_model.pkl"

FEATURE_ORDER: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def _model_path() -> Path:
    raw = os.environ.get("MODEL_PATH")
    return Path(raw).expanduser() if raw else _DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def _training_medians() -> dict[str, float]:
    df = pd.read_csv(_DATA_CSV)
    cols = ["SkinThickness", "Insulin", "BMI"]
    for col in cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    medians: dict[str, float] = {}
    for name in FEATURE_ORDER:
        if name in df.columns:
            medians[name] = float(df[name].median())
    return medians


@lru_cache(maxsize=1)
def _load_model():
    path = _model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `uv run model` to train and save it."
        )
    return joblib.load(path)


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    h_m = height_cm / 100.0
    return float(weight_kg / (h_m * h_m))


def _risk_class(score: float) -> Literal["low", "moderate", "high"]:
    if score < 34:
        return "low"
    if score < 67:
        return "moderate"
    return "high"


def predict_risk(payload: OnboardingIn) -> tuple[float, Literal["low", "moderate", "high"]]:
    """Return risk score 0–100 and class from the XGBoost model's positive-class probability."""
    med = _training_medians()
    bmi = compute_bmi(payload.height_cm, payload.weight_kg)
    row = {
        "Pregnancies": 0.0,
        "Glucose": float(payload.glucose),
        "BloodPressure": float(payload.blood_pressure_diastolic),
        "SkinThickness": med.get("SkinThickness", 29.0),
        "Insulin": med.get("Insulin", 80.0),
        "BMI": bmi,
        "DiabetesPedigreeFunction": med.get("DiabetesPedigreeFunction", 0.5),
        "Age": float(payload.age),
    }
    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    model = _load_model()
    if not hasattr(model, "predict_proba"):
        pred = float(model.predict(X)[0])
        score = pred * 100.0
        return score, _risk_class(score)
    proba = model.predict_proba(X)[0]
    positive_idx = 1 if proba.shape[0] > 1 else 0
    score = float(proba[positive_idx] * 100.0)
    return score, _risk_class(score)

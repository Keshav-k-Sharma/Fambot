from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib

from fambot_backend.cardio_features import (
    build_feature_frame,
    cholesterol_mg_dl_to_ordinal,
    glucose_mg_dl_to_ordinal,
)
from fambot_backend.schemas import OnboardingIn
from fambot_backend.services.family_risk_aggregate import (
    compute_family_risk_feature_row,
    neutral_family_features,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / "cardiovascular_model.pkl"


def _model_path() -> Path:
    raw = os.environ.get("MODEL_PATH")
    return Path(raw).expanduser() if raw else _DEFAULT_MODEL_PATH


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


def predict_risk(
    payload: OnboardingIn,
    *,
    subject_uid: str | None = None,
) -> tuple[float, Literal["low", "moderate", "high"]]:
    """Return risk score 0–100 and class from the pipeline's positive-class probability."""
    fam = (
        compute_family_risk_feature_row(subject_uid)
        if subject_uid
        else neutral_family_features()
    )
    X = build_feature_frame(
        age=payload.age,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        blood_pressure_systolic=payload.blood_pressure_systolic,
        blood_pressure_diastolic=payload.blood_pressure_diastolic,
        gender=payload.gender,
        cholesterol=cholesterol_mg_dl_to_ordinal(payload.cholesterol_mg_dl),
        gluc_ordinal=glucose_mg_dl_to_ordinal(payload.glucose_mg_dl),
        smokes=payload.smokes,
        drinks_alcohol=payload.drinks_alcohol,
        physically_active=payload.physically_active,
        **fam,
    )

    model = _load_model()
    if not hasattr(model, "predict_proba"):
        pred = float(model.predict(X)[0])
        score = pred * 100.0
        return score, _risk_class(score)
    proba = model.predict_proba(X)[0]
    positive_idx = 1 if proba.shape[0] > 1 else 0
    score = float(proba[positive_idx] * 100.0)
    return score, _risk_class(score)

"""Shared cardiovascular model feature layout — must match training in model.py."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

CholesterolOrdinal = Literal[1, 2, 3]
GlucOrdinal = Literal[1, 2, 3]

# Raw columns from cardio_train (after age -> age_years); order before derived features.
BASE_FEATURES: list[str] = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

DERIVED_FEATURES: list[str] = ["bmi", "pulse_pressure", "map_approx"]

# Optional family-group aggregates (member app risk scores, relationship-weighted).
FAMILY_FEATURES: list[str] = [
    "fam_weighted_mean_risk",
    "fam_max_member_risk",
    "fam_first_deg_mean_risk",
    "fam_any_member_high_risk",
]

# Full column order for the sklearn Pipeline / DataFrame row.
CORE_FEATURE_ORDER: list[str] = BASE_FEATURES + DERIVED_FEATURES
FEATURE_ORDER: list[str] = CORE_FEATURE_ORDER + FAMILY_FEATURES


class Gender(str, Enum):
    """API-facing gender; maps to cardio_train codes (1 = female, 2 = male)."""

    female = "female"
    male = "male"


def gender_to_dataset_code(gender: Gender | str) -> int:
    if isinstance(gender, str):
        gender = Gender(gender)
    if gender == Gender.female:
        return 1
    if gender == Gender.male:
        return 2
    raise ValueError(f"Unsupported gender: {gender!r}")


def cholesterol_mg_dl_to_ordinal(mg_dl: float) -> CholesterolOrdinal:
    """Map fasting total cholesterol (mg/dL) to cardio_train ordinal 1–3.

    Bands align with common ATP III-style total-cholesterol cutoffs:
    1 = < 200 (desirable), 2 = 200–239 (borderline high), 3 = ≥ 240 (high).
    """
    value = float(mg_dl)
    if value < 200.0:
        return 1
    if value < 240.0:
        return 2
    return 3


def glucose_mg_dl_to_ordinal(mg_dl: float) -> GlucOrdinal:
    """Map fasting plasma glucose (mg/dL, ≥ 8 h fast) to cardio_train `gluc` ordinal 1–3.

    Bands align with ADA fasting glucose categories:
    1 = < 100 (normal), 2 = 100–125 (above normal), 3 = ≥ 126 (well above normal).
    """
    value = float(mg_dl)
    if value < 100.0:
        return 1
    if value < 126.0:
        return 2
    return 3


def _optional_bool_to_float(v: bool | None) -> float:
    if v is None:
        return float(np.nan)
    return 1.0 if v else 0.0


def _optional_float_feature(v: float | None) -> float:
    if v is None:
        return float(np.nan)
    return float(v)


def build_feature_frame(
    *,
    age: int,
    height_cm: float,
    weight_kg: float,
    blood_pressure_systolic: float,
    blood_pressure_diastolic: float,
    gender: Gender | str,
    cholesterol: int,
    gluc_ordinal: int,
    smokes: bool | None = None,
    drinks_alcohol: bool | None = None,
    physically_active: bool | None = None,
    fam_weighted_mean_risk: float | None = None,
    fam_max_member_risk: float | None = None,
    fam_first_deg_mean_risk: float | None = None,
    fam_any_member_high_risk: float | None = None,
) -> pd.DataFrame:
    """One-row DataFrame matching FEATURE_ORDER for the saved cardiovascular Pipeline."""
    if blood_pressure_systolic <= blood_pressure_diastolic:
        raise ValueError(
            "Systolic blood pressure must be greater than diastolic blood pressure."
        )

    h_m = float(height_cm) / 100.0
    w = float(weight_kg)
    ap_hi = float(blood_pressure_systolic)
    ap_lo = float(blood_pressure_diastolic)

    bmi = w / (h_m * h_m)
    pulse_pressure = ap_hi - ap_lo
    map_approx = (ap_hi + 2.0 * ap_lo) / 3.0

    row: dict[str, float] = {
        "age_years": float(age),
        "gender": float(gender_to_dataset_code(gender)),
        "height": float(height_cm),
        "weight": w,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": float(cholesterol),
        "gluc": float(gluc_ordinal),
        "smoke": _optional_bool_to_float(smokes),
        "alco": _optional_bool_to_float(drinks_alcohol),
        "active": _optional_bool_to_float(physically_active),
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "map_approx": map_approx,
        "fam_weighted_mean_risk": _optional_float_feature(fam_weighted_mean_risk),
        "fam_max_member_risk": _optional_float_feature(fam_max_member_risk),
        "fam_first_deg_mean_risk": _optional_float_feature(fam_first_deg_mean_risk),
        "fam_any_member_high_risk": _optional_float_feature(fam_any_member_high_risk),
    }

    return pd.DataFrame([row], columns=FEATURE_ORDER)

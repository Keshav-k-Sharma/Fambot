"""Pydantic schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fambot_backend.cardio_features import Gender
from fambot_backend.schemas import OnboardingIn, SignupIn


@pytest.mark.unit
def test_signup_name_stripped() -> None:
    s = SignupIn(email="a@b.co", password="secret1", name="  Ada  ")
    assert s.name == "Ada"


@pytest.mark.unit
def test_signup_name_empty_rejected() -> None:
    with pytest.raises(ValidationError):
        SignupIn(email="a@b.co", password="secret1", name="   ")


@pytest.mark.unit
def test_onboarding_bp_bounds() -> None:
    OnboardingIn(
        age=40,
        height_cm=170,
        weight_kg=70,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        gender=Gender.male,
        cholesterol_mg_dl=180,
        glucose_mg_dl=90,
    )
    with pytest.raises(ValidationError):
        OnboardingIn(
            age=40,
            height_cm=170,
            weight_kg=70,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            gender=Gender.male,
            cholesterol_mg_dl=10,  # below ge=50
            glucose_mg_dl=90,
        )


@pytest.mark.unit
def test_onboarding_accepts_lab_mg_dl() -> None:
    payload = OnboardingIn(
        age=40,
        height_cm=170,
        weight_kg=70,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        gender=Gender.female,
        cholesterol_mg_dl=220.5,
        glucose_mg_dl=112.0,
    )
    assert payload.cholesterol_mg_dl == 220.5
    assert payload.glucose_mg_dl == 112.0

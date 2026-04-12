from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=4096)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=4096)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    uid: str
    email: str | None = None


class OnboardingIn(BaseModel):
    age: int = Field(ge=1, le=120)
    height_cm: float = Field(ge=50, le=260, description="Height in centimeters")
    weight_kg: float = Field(ge=20, le=400, description="Weight in kilograms")
    glucose: float = Field(ge=1, le=600, description="Plasma glucose concentration")
    blood_pressure_diastolic: float = Field(
        ge=20,
        le=200,
        description="Diastolic blood pressure (mm Hg); matches Pima BloodPressure column",
    )
    blood_pressure_systolic: float | None = Field(
        default=None,
        ge=40,
        le=300,
        description="Optional systolic BP for storage/display",
    )


class UserProfileOut(BaseModel):
    uid: str
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    glucose: float | None = None
    blood_pressure_systolic: float | None = None
    blood_pressure_diastolic: float | None = None
    bmi: float | None = None
    risk_score: float | None = Field(
        default=None,
        description="0–100 probability-style score from the diabetes risk model",
    )
    risk_class: Literal["low", "moderate", "high"] | None = None
    onboarding_complete: bool = False
    updated_at: datetime | None = None


class OnboardingOut(BaseModel):
    profile: UserProfileOut
    risk_score: float
    risk_class: Literal["low", "moderate", "high"]

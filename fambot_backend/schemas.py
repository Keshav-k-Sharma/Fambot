from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from fambot_backend.cardio_features import Gender


class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=4096)
    name: str = Field(max_length=128, description="Display name; stored in Firebase Auth and Firestore")

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("name cannot be empty")
        return s


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=4096)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    uid: str
    email: str | None = None


CholesterolLevel = Literal[1, 2, 3]
GlucoseLevel = Literal[1, 2, 3]


class OnboardingIn(BaseModel):
    """Vitals and survey fields aligned with the cardiovascular training CSV (cardio_train)."""

    age: int = Field(ge=1, le=120, description="Age in years")
    height_cm: float = Field(ge=120, le=220, description="Height in centimeters")
    weight_kg: float = Field(ge=35, le=250, description="Weight in kilograms")
    blood_pressure_systolic: float = Field(
        ge=80,
        le=250,
        description="Systolic blood pressure (mm Hg)",
    )
    blood_pressure_diastolic: float = Field(
        ge=40,
        le=150,
        description="Diastolic blood pressure (mm Hg)",
    )
    gender: Gender = Field(description='"female" or "male" (maps to dataset codes 1 / 2)')
    cholesterol: CholesterolLevel = Field(
        description=(
            "1 = normal, 2 = above normal, 3 = well above normal (ordinal from cardio_train)"
        ),
    )
    glucose_level: GlucoseLevel = Field(
        description=(
            "1 = normal, 2 = above normal, 3 = well above normal (blood glucose category, ordinal)"
        ),
    )
    smokes: bool | None = Field(
        default=None,
        description="Whether the user smokes; omit to impute from training distribution",
    )
    drinks_alcohol: bool | None = Field(
        default=None,
        description="Whether the user drinks alcohol; omit to impute",
    )
    physically_active: bool | None = Field(
        default=None,
        description="Whether the user is physically active; omit to impute",
    )


class UserProfileOut(BaseModel):
    uid: str
    display_name: str | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    gender: Gender | None = None
    cholesterol: int | None = None
    glucose_level: int | None = None
    smokes: bool | None = None
    drinks_alcohol: bool | None = None
    physically_active: bool | None = None
    blood_pressure_systolic: float | None = None
    blood_pressure_diastolic: float | None = None
    bmi: float | None = None
    risk_score: float | None = Field(
        default=None,
        description="0–100 score from cardiovascular risk model (positive-class probability × 100)",
    )
    risk_class: Literal["low", "moderate", "high"] | None = None
    onboarding_complete: bool = False
    updated_at: datetime | None = None


class OnboardingOut(BaseModel):
    profile: UserProfileOut
    risk_score: float
    risk_class: Literal["low", "moderate", "high"]


class RiskOut(BaseModel):
    """Stored cardiovascular risk from the last successful onboarding (Firestore)."""

    risk_score: float
    risk_class: Literal["low", "moderate", "high"]

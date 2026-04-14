from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal

from firebase_admin import firestore

from fambot_backend.cardio_features import Gender
from fambot_backend.core.firebase_init import init_firebase
from fambot_backend.schemas import OnboardingIn, UserProfileOut


def _db():
    init_firebase()
    return firestore.client()


def _parse_gender(raw: Any) -> Gender | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw in ("female", "male"):
        return Gender(raw)
    return None


def _doc_to_profile(uid: str, data: dict[str, Any] | None) -> UserProfileOut:
    if not data:
        return UserProfileOut(uid=uid, onboarding_complete=False)
    ts = data.get("updatedAt")
    updated: datetime | None = None
    if isinstance(ts, datetime):
        updated = ts
    return UserProfileOut(
        uid=uid,
        display_name=data.get("displayName"),
        age=data.get("age"),
        height_cm=data.get("heightCm"),
        weight_kg=data.get("weightKg"),
        gender=_parse_gender(data.get("gender")),
        cholesterol=data.get("cholesterol"),
        gluc_ordinal=data.get("glucOrdinal") if data.get("glucOrdinal") is not None else data.get("glucoseLevel"),
        smokes=data.get("smokes"),
        drinks_alcohol=data.get("drinksAlcohol"),
        physically_active=data.get("physicallyActive"),
        blood_pressure_systolic=data.get("bloodPressureSystolic"),
        blood_pressure_diastolic=data.get("bloodPressureDiastolic"),
        bmi=data.get("bmi"),
        risk_score=data.get("riskScore"),
        risk_class=data.get("riskClass"),
        onboarding_complete=bool(data.get("onboardingComplete", False)),
        updated_at=updated,
    )


def ensure_user_document(uid: str, *, display_name: str | None = None) -> None:
    """Create a minimal Firestore user doc if missing (signup)."""
    if os.environ.get("FAMBOT_SKIP_FIRESTORE") == "1":
        return
    ref = _db().collection("users").document(uid)
    if ref.get().exists:
        return
    doc: dict[str, Any] = {"onboardingComplete": False}
    if display_name:
        doc["displayName"] = display_name
    ref.set(doc)


def get_user_profile(uid: str) -> UserProfileOut:
    if os.environ.get("FAMBOT_SKIP_FIRESTORE") == "1":
        return UserProfileOut(uid=uid, onboarding_complete=False)
    snap = _db().collection("users").document(uid).get()
    return _doc_to_profile(uid, snap.to_dict())


def upsert_onboarding(
    uid: str,
    payload: OnboardingIn,
    bmi: float,
    risk_score: float,
    risk_class: Literal["low", "moderate", "high"],
) -> UserProfileOut:
    if os.environ.get("FAMBOT_SKIP_FIRESTORE") == "1":
        now = datetime.now(timezone.utc)
        return UserProfileOut(
            uid=uid,
            age=payload.age,
            height_cm=payload.height_cm,
            weight_kg=payload.weight_kg,
            gender=payload.gender,
            cholesterol=payload.cholesterol,
            gluc_ordinal=payload.gluc_ordinal,
            smokes=payload.smokes,
            drinks_alcohol=payload.drinks_alcohol,
            physically_active=payload.physically_active,
            blood_pressure_systolic=payload.blood_pressure_systolic,
            blood_pressure_diastolic=payload.blood_pressure_diastolic,
            bmi=bmi,
            risk_score=risk_score,
            risk_class=risk_class,
            onboarding_complete=True,
            updated_at=now,
        )
    ref = _db().collection("users").document(uid)
    now = datetime.now(timezone.utc)
    update = {
        "age": payload.age,
        "heightCm": payload.height_cm,
        "weightKg": payload.weight_kg,
        "gender": payload.gender.value,
        "cholesterol": payload.cholesterol,
        "glucOrdinal": payload.gluc_ordinal,
        "smokes": payload.smokes,
        "drinksAlcohol": payload.drinks_alcohol,
        "physicallyActive": payload.physically_active,
        "bloodPressureSystolic": payload.blood_pressure_systolic,
        "bloodPressureDiastolic": payload.blood_pressure_diastolic,
        "bmi": bmi,
        "riskScore": risk_score,
        "riskClass": risk_class,
        "onboardingComplete": True,
        "updatedAt": now,
    }
    ref.set(update, merge=True)
    return _doc_to_profile(uid, {**update, "updatedAt": now})

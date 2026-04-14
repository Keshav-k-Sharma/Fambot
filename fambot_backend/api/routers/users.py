from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import (
    OnboardingIn,
    OnboardingOut,
    RiskOut,
    UserProfileOut,
)
from fambot_backend.services.firestore_users import get_user_profile, upsert_onboarding
from fambot_backend.services.inference import compute_bmi, predict_risk

router = APIRouter(prefix="/me", tags=["me"])


@router.get("", response_model=UserProfileOut)
def read_me(uid: str = Depends(firebase_uid)) -> UserProfileOut:
    return get_user_profile(uid)


@router.get("/risk", response_model=RiskOut)
def read_me_risk(uid: str = Depends(firebase_uid)) -> RiskOut:
    profile = get_user_profile(uid)
    if (
        not profile.onboarding_complete
        or profile.risk_score is None
        or profile.risk_class is None
    ):
        raise HTTPException(
            status_code=404,
            detail="Risk score not available; complete onboarding first.",
        )
    return RiskOut(risk_score=profile.risk_score, risk_class=profile.risk_class)


@router.put("/onboarding", response_model=OnboardingOut)
def complete_onboarding(
    body: OnboardingIn,
    uid: str = Depends(firebase_uid),
) -> OnboardingOut:
    try:
        score, rclass = predict_risk(body)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    bmi = compute_bmi(body.height_cm, body.weight_kg)
    profile = upsert_onboarding(uid, body, bmi, score, rclass)
    return OnboardingOut(profile=profile, risk_score=score, risk_class=rclass)

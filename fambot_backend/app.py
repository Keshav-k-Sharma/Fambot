from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError

from fambot_backend.deps import firebase_uid
from fambot_backend.firebase_init import init_firebase
from fambot_backend.firestore_users import ensure_user_document, get_user_profile, upsert_onboarding
from fambot_backend.identity_toolkit import IdentityToolkitError, sign_in_with_password
from fambot_backend.inference import compute_bmi, predict_risk
from fambot_backend.jwt_tokens import mint_access_token
from fambot_backend.schemas import (
    LoginIn,
    OnboardingIn,
    OnboardingOut,
    SignupIn,
    TokenOut,
    UserProfileOut,
)


def _identity_toolkit_http(exc: IdentityToolkitError) -> HTTPException:
    m = exc.message
    if exc.status_code == 400 and any(
        s in m
        for s in (
            "INVALID_PASSWORD",
            "EMAIL_NOT_FOUND",
            "INVALID_EMAIL",
            "INVALID_LOGIN_CREDENTIALS",
        )
    ):
        return HTTPException(status_code=401, detail="Invalid email or password")
    if "USER_DISABLED" in m:
        return HTTPException(status_code=403, detail="Account disabled")
    if exc.status_code >= 500:
        return HTTPException(status_code=502, detail="Authentication service error")
    return HTTPException(status_code=400, detail=m)

app = FastAPI(title="Fambot API", version="0.2.0")

_origins = os.environ.get("FAMBOT_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/auth/signup", response_model=TokenOut)
def auth_signup(body: SignupIn) -> TokenOut:
    init_firebase()
    try:
        user = auth.create_user(email=str(body.email), password=body.password)
    except auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=409, detail="Email already registered") from None
    except FirebaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_user_document(user.uid)
    try:
        token, exp = mint_access_token(user.uid, str(body.email))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TokenOut(
        access_token=token,
        expires_in=exp,
        uid=user.uid,
        email=str(body.email),
    )


@app.post("/v1/auth/login", response_model=TokenOut)
def auth_login(body: LoginIn) -> TokenOut:
    try:
        data = sign_in_with_password(str(body.email), body.password)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except IdentityToolkitError as exc:
        raise _identity_toolkit_http(exc) from exc
    uid = data.get("localId")
    email = data.get("email")
    if not uid or not isinstance(uid, str):
        raise HTTPException(status_code=502, detail="Invalid auth response")
    try:
        token, exp = mint_access_token(uid, email if isinstance(email, str) else None)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TokenOut(access_token=token, expires_in=exp, uid=uid, email=email if isinstance(email, str) else None)


@app.get("/v1/me", response_model=UserProfileOut)
def read_me(uid: str = Depends(firebase_uid)) -> UserProfileOut:
    return get_user_profile(uid)


@app.put("/v1/me/onboarding", response_model=OnboardingOut)
def complete_onboarding(
    body: OnboardingIn,
    uid: str = Depends(firebase_uid),
) -> OnboardingOut:
    score, rclass = predict_risk(body)
    bmi = compute_bmi(body.height_cm, body.weight_kg)
    profile = upsert_onboarding(uid, body, bmi, score, rclass)
    return OnboardingOut(profile=profile, risk_score=score, risk_class=rclass)


def run() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("fambot_backend.app:app", host=host, port=port, reload=False)

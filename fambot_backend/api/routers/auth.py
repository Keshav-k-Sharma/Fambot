from __future__ import annotations

from fastapi import APIRouter, HTTPException
from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError

from fambot_backend.core.firebase_init import init_firebase
from fambot_backend.core.jwt_tokens import mint_access_token
from fambot_backend.schemas import LoginIn, SignupIn, TokenOut
from fambot_backend.services.firestore_users import ensure_user_document
from fambot_backend.services.identity_toolkit import IdentityToolkitError, sign_in_with_password

router = APIRouter(prefix="/auth", tags=["auth"])


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


@router.post("/signup", response_model=TokenOut)
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


@router.post("/login", response_model=TokenOut)
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

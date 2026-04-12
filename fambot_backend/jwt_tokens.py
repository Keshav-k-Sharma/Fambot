from __future__ import annotations

import os
from datetime import datetime, timezone

import jwt

_DEFAULT_EXPIRES = 3600


def _secret() -> str:
    s = os.environ.get("FAMBOT_JWT_SECRET", "").strip()
    if not s:
        raise ValueError("FAMBOT_JWT_SECRET is not set")
    return s


def expires_seconds() -> int:
    raw = os.environ.get("FAMBOT_JWT_EXPIRES_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_EXPIRES
    return max(60, int(raw))


def mint_access_token(uid: str, email: str | None = None) -> tuple[str, int]:
    """Return (jwt, expires_in_seconds)."""
    exp_s = expires_seconds()
    now_ts = int(datetime.now(timezone.utc).timestamp())
    payload: dict = {
        "sub": uid,
        "exp": now_ts + exp_s,
        "iat": now_ts,
    }
    if email:
        payload["email"] = email
    token = jwt.encode(payload, _secret(), algorithm="HS256")
    return token, exp_s


def decode_and_verify(token: str) -> dict:
    """Verify HS256 JWT and return claims (includes sub, exp, email)."""
    return jwt.decode(token, _secret(), algorithms=["HS256"])

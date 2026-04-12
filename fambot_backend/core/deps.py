from __future__ import annotations

import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from fambot_backend.core.jwt_tokens import decode_and_verify

_bearer = HTTPBearer(auto_error=False)


async def firebase_uid(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if os.environ.get("FAMBOT_SKIP_AUTH") == "1":
        # Local-only escape hatch; never enable in production.
        return os.environ.get("FAMBOT_DEV_UID", "dev-user")
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    try:
        payload = decode_and_verify(creds.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: FAMBOT_JWT_SECRET is not set",
        ) from exc
    except jwt.exceptions.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired access token") from exc
    uid = payload.get("sub")
    if not uid or not isinstance(uid, str):
        raise HTTPException(status_code=401, detail="Token missing subject")
    return uid

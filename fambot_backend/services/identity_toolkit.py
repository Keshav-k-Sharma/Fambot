from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _api_key() -> str:
    k = os.environ.get("FIREBASE_WEB_API_KEY", "").strip()
    if not k:
        raise ValueError("FIREBASE_WEB_API_KEY is not set")
    return k


def sign_in_with_password(email: str, password: str) -> dict[str, Any]:
    """
    Call Identity Toolkit signInWithPassword.
    Returns the JSON body on success (includes localId, idToken, email, expiresIn, refreshToken).
    Raises IdentityToolkitError with .status_code and .message on failure.
    """
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={_api_key()}"
    body = json.dumps(
        {"email": email, "password": password, "returnSecureToken": True}
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise IdentityToolkitError(exc.code, raw) from exc
        err = data.get("error") or {}
        message = err.get("message", raw)
        raise IdentityToolkitError(exc.code, message) from exc


class IdentityToolkitError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)

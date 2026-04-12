from __future__ import annotations

import os

import firebase_admin
from firebase_admin import credentials


def init_firebase() -> None:
    """Initialize the default Firebase app once (Firestore + Auth verification)."""
    if firebase_admin._apps:
        return
    # Application Default Credentials (Cloud Run, GCE) or GOOGLE_APPLICATION_CREDENTIALS locally.
    cred = credentials.ApplicationDefault()
    project_id = os.environ.get("FIREBASE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    opts: dict = {}
    if project_id:
        opts["projectId"] = project_id
    firebase_admin.initialize_app(cred, opts)

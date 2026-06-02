# Agent guide: Fambot Backend

Instructions for AI coding agents and human contributors working in this repository. Read this before making non-trivial changes.

---

## Documentation maintenance

Whenever you add or materially change a **feature** (new or altered endpoints, environment variables, authentication, request/response shapes, Firestore fields, or training/inference contracts), **update the docs in the same effort**:

- **[`README.md`](README.md)** — user-facing overview, env reference, and API reference.
- **[`.env.example`](.env.example)** — template for local configuration; copy to **`.env`** (gitignored). The app loads `.env` at import time via `python-dotenv` in [`fambot_backend/app.py`](fambot_backend/app.py).
- **This file (`AGENTS.md`)** — when conventions, directory layout, auth flow, or “what to edit for task X” guidance changes.

The repository should remain the **source of truth**; avoid shipping behavior without documenting it.

---

## Project purpose

**Fambot Backend** is a Python service that scores **cardiovascular** risk from a pipeline trained on `cardio_train.csv`. It is **not** a diabetes-specific or lab-glucose (mg/dL) product; survey ordinals mirror the dataset (e.g. JSON `gluc_ordinal` → feature `gluc`). It:

1. Exposes a **FastAPI** HTTP API with **JWT Bearer** authentication (tokens minted by this service after email/password signup/login against Firebase).
2. Persists user onboarding and profile fields in **Google Cloud Firestore** (`users/{uid}`).
3. Loads a **joblib-serialized sklearn Pipeline** (`cardiovascular_model.pkl`) trained from `model.py` on `sources/cardio_train.csv`, and converts onboarding input into a feature row compatible with that pipeline (see `fambot_backend/cardio_features.py`).

Agents should preserve the separation between **training** (`model.py`), **HTTP layer** (`fambot_backend/app.py` and `fambot_backend/api/routers/`), **inference** (`fambot_backend/services/inference.py`), **persistence** (`fambot_backend/services/firestore_users.py`), and **family/invites** (`fambot_backend/services/family_invites.py`, `fambot_backend/services/family_roles.py`).

---

## Tech stack (authoritative)

- **Python** `>= 3.14` per `pyproject.toml`.
- **Package manager:** `uv` is the assumed workflow (`uv sync`, `uv run …`).
- **Web:** FastAPI + Uvicorn.
- **Google:** `firebase-admin` (Admin Auth user creation, Firestore).
- **Auth:** `PyJWT` (HS256 access tokens); Identity Toolkit REST for password login (see `fambot_backend/services/identity_toolkit.py`).
- **ML:** pandas, scikit-learn, xgboost, joblib, matplotlib (training only).

---

## Entry points and scripts

| Command | Defined in | What it runs |
|---------|------------|----------------|
| `uv run api` | `[project.scripts] api = "fambot_backend.app:run"` | Uvicorn on `fambot_backend.app:app`. |
| `uv run model` | `[project.scripts] model = "model:main"` | Training pipeline in `model.py`; writes `cardiovascular_model.pkl`, optional `cardiovascular_model.threshold.json`, and `feature_importance.png`. |
| `bash scripts/render_start.sh` | Shell (Render) | Writes `GOOGLE_SERVICE_ACCOUNT_JSON` to a temp file, sets `GOOGLE_APPLICATION_CREDENTIALS`, `exec uv run api`. |

---

## Directory map

```
fambot_backend/
  app.py                    # FastAPI app, CORS, include_router, uvicorn runner
  cardio_features.py        # FEATURE_ORDER shared with model.py; build_feature_frame
  schemas.py                # Pydantic models (API JSON shape)
  api/routers/              # health, auth, users, invitations, documents (/documents/*), chats (/chat*, /chats)
  core/
    deps.py                 # HTTPBearer → JWT → firebase_uid (Firebase Auth uid string)
    jwt_tokens.py           # Mint / verify access tokens (FAMBOT_JWT_SECRET)
    firebase_init.py        # firebase_admin.initialize_app (ADC)
    chat_orchestrator.py    # Unified chat turn lifecycle, SSE eventing, persistence finalization
    context_builder.py      # Chat context slicing (history window)
    tool_runtime.py         # Tool dispatch bridge for model tool calls
  providers/
    model_provider.py       # Provider interface for model streaming/completion
    gemini_provider.py      # Gemini implementation with streaming + tool loop
  persistence/
    chat_repository.py      # Chat/message/turn persistence + idempotency handling
  telemetry/
    chat_telemetry.py       # Structured logs/metrics for chat turn lifecycle
  services/
    inference.py            # MODEL_PATH, joblib load, predict_risk
    identity_toolkit.py     # signInWithPassword for POST /auth/login
    firestore_users.py      # get_user_profile, upsert_onboarding, ensure_user_document, familyGroupId helpers
    document_storage.py     # upload/list/get/delete reports in Firebase Storage (documents/{uid}/...)
    gemini_document_analysis.py  # Gemini chat+analysis orchestration with profile/doc context
    chat_history.py         # Firestore-backed chat sessions/messages under users/{uid}/chats/*
    family_invites.py       # family groups, invites, QR payload, accept/remove flows
    family_roles.py         # reciprocal family role mapping (vocabulary)
    family_risk_aggregate.py # peer risk scores + relationship weights for model features
model.py                    # Offline training; LR vs XGB vs HistGradientBoosting; saves champion
sources/cardio_train.csv    # Training data (semicolon-separated)
render.yaml                 # Render Blueprint (build/start, env var names)
scripts/render_start.sh     # Render: materialize GOOGLE_SERVICE_ACCOUNT_JSON → ADC file, start API
```

---

## Deployment (Render)

Production on **Render** is documented in **[`README.md`](README.md)** (Firebase checklist, `render.yaml`, `scripts/render_start.sh`, required secrets). On Render, credentials are supplied via `GOOGLE_SERVICE_ACCOUNT_JSON` and the start script; local dev typically uses `GOOGLE_APPLICATION_CREDENTIALS` pointing at a file path.

---

## Environment variables agents must respect

| Variable | Agent-relevant behavior |
|----------|-------------------------|
| `MODEL_PATH` | Overrides default `cardiovascular_model.pkl` path in `services.inference._model_path()`. |
| `FIREBASE_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` | Passed into Firebase app options when set. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Local/service credential path for ADC. On Render, set by `scripts/render_start.sh` after writing `GOOGLE_SERVICE_ACCOUNT_JSON`. |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Render: full service account JSON in one env var; consumed only by `scripts/render_start.sh`. |
| `FAMBOT_SKIP_AUTH=1` | **Dev only.** Bypasses JWT verification; uses `FAMBOT_DEV_UID`. Never enable in production code paths or docs that imply production. |
| `FAMBOT_DEV_UID` | UID string when `FAMBOT_SKIP_AUTH=1` (default `dev-user`). |
| `FAMBOT_SKIP_FIRESTORE=1` | Skips Firestore; returns in-memory profile for testing. |
| `FAMBOT_JWT_SECRET` | Required (when auth is not skipped) to sign tokens at signup/login and verify them on protected routes. |
| `FAMBOT_JWT_EXPIRES_SECONDS` | Access token TTL (defaults documented in `core/jwt_tokens.py` / README). |
| `FIREBASE_WEB_API_KEY` | Required for `POST /auth/login` (Identity Toolkit). |
| `FIREBASE_STORAGE_BUCKET` | Required for report uploads to Firebase Storage. |
| `GEMINI_API_KEY` | Required for document analysis (`POST /documents` with `analyze=true`, or `POST /documents/{doc_id}/analyze`) and **Gemini chat** (custom function tools; optional File Search indexing on upload). |
| `GEMINI_REPORT_MODEL` | Optional Gemini model override (default `gemini-2.5-flash`). |
| `FAMBOT_GEMINI_DISABLE_FILE_SEARCH` | Set to `1` to disable **Gemini File Search** store creation and upload indexing. Chat turns do not attach built-in File Search while using custom function tools, because Gemini rejects combining those tool types in one request. |
| `FAMBOT_CORS_ORIGINS` | Comma-separated origins; default allows `*`. |
| `FAMBOT_FAMILY_INVITE_TTL_SECONDS` | Family invite token TTL (default 86400; clamped 60–2592000). |
| `FAMBOT_INVITE_BASE_URL` | Optional prefix for invite URLs embedded in QR codes; if unset, `fambot://family-invite?token=…` is used. |
| `FAMBOT_RUN_EXTERNAL_TESTS` | Set to `1` to enable pytest `external` tests that call real HTTP APIs (Identity Toolkit, optional health URL). |
| `FAMBOT_RUN_GEMINI_LIVE` | Set to `1` with `GEMINI_API_KEY` to enable pytest `gemini_live` tests (one minimal real `generate_content` call; network cost). |
| `FAMBOT_EXTERNAL_TEST_EMAIL` / `FAMBOT_EXTERNAL_TEST_PASSWORD` | Optional disposable Firebase user for the optional successful-login external test. |
| `FAMBOT_EXTERNAL_HEALTH_URL` | Optional URL for a trivial GET in external smoke tests. |

When adding tests or local scripts, prefer `FAMBOT_SKIP_*` flags over mocking unless the test specifically targets Firebase.

---

## API and data conventions

1. **JSON API (Pydantic):** `snake_case` field names (`height_cm`, `blood_pressure_diastolic`, …).
2. **Firestore documents:** `camelCase` keys (`displayName`, `heightCm`, `bloodPressureDiastolic`, …). Mapping lives in `services/firestore_users.py` only—keep it centralized.
3. **Model feature names:** Must match `FEATURE_ORDER` in `fambot_backend/cardio_features.py` (shared with `model.py`) and the column order expected by the saved pipeline. Any new user-facing field that maps into the model requires coordinated updates in:
   - `schemas.OnboardingIn`
   - `cardio_features.build_feature_frame` (and `model.py` training if columns change)
   - Optionally `model.py` if training data or preprocessing changes

---

## Authentication flow

1. **Signup:** `POST /auth/signup` creates a Firebase Auth user with `firebase_admin.auth.create_user` (email, password, required `name` as `display_name`), ensures a Firestore user doc, mints a JWT (`sub` = Firebase `uid`).
2. **Login:** `POST /auth/login` calls Identity Toolkit `signInWithPassword` (requires `FIREBASE_WEB_API_KEY`), then mints the same JWT shape from `localId`.
3. **Protected routes:** client sends `Authorization: Bearer <JWT>`. `core.deps.firebase_uid` verifies the JWT with `core.jwt_tokens.decode_and_verify` and returns `sub` as the uid string (name kept for minimal churn).
4. If `FAMBOT_SKIP_AUTH=1`, returns a fixed dev UID without validating a token.

Agents adding new protected routes should use `uid: str = Depends(firebase_uid)` consistently.

---

## Inference contract

- `predict_risk` builds a **single-row** `pandas.DataFrame` with columns exactly `FEATURE_ORDER` from `cardio_features.py` (core vitals plus optional **family-group aggregate** columns: weighted mean / max / first-degree mean of peers’ stored `risk_score`, and a binary-style flag for any peer in the high score band).
- Optional lifestyle fields omitted in the API are passed as **missing values** and imputed by the **fitted `SimpleImputer`** inside the saved pipeline.
- Risk score is derived from `predict_proba` positive class × 100 when available.
- `compute_bmi` is used for both persistence; the feature row also includes derived BMI, pulse pressure, and MAP proxy from height/weight/BP.

Breaking changes to the saved pipeline (e.g. different column set) require retraining and redeploying `cardiovascular_model.pkl`.

---

## Training contract (`model.py`)

- Uses stratified train/test split and 5-fold CV.
- Champion is chosen by **CV ROC-AUC** among logistic regression, **RandomizedSearchCV** XGBoost, and **HistGradientBoostingClassifier** random search.
- Saves the **entire fitted Pipeline** with `joblib.dump` so inference can call `predict_proba` on the same structure.

If you change preprocessing in `model.py`, keep `fambot_backend/cardio_features.py` in sync or retrain and ship a new artifact.

---

## Git and generated artifacts

`.gitignore` excludes:

- `.env` (local secrets)
- `.venv`
- `cardiovascular_model.threshold.json`
- `feature_importance.png`
- `firebase-admin.json` (local Firebase Admin SDK key; never commit)

The API **fails at runtime** if `cardiovascular_model.pkl` is missing (unless you point `MODEL_PATH` to a valid file). **Render / production:** commit `cardiovascular_model.pkl` after training locally; do not rely on `uv run model` during the hosting provider’s build (slow and unnecessary). Agents should mention `uv run model` for local/CI training when adding features that require the model.

---

## CORS

Configured in `app.py` via `FAMBOT_CORS_ORIGINS` (comma-separated). Default is permissive (`*`). Tighten for known web app origins in production.

---

## What to edit for common tasks

| Task | Primary files |
|------|----------------|
| New endpoint | `api/routers/`, `app.py` (include router), possibly `schemas.py`; update **README** API section |
| Change request validation | `schemas.py` |
| Change Firestore fields | `services/firestore_users.py`, `schemas.py` (read/write models) |
| Family invites / roles | `services/family_invites.py`, `services/family_roles.py`, `services/family_risk_aggregate.py`, `api/routers/invitations.py`, `schemas.py` |
| Report upload + retrieval | `api/routers/documents.py`, `services/document_storage.py`, `core/firebase_init.py`, `schemas.py` |
| Document analyze (Gemini + profile) | `api/routers/documents.py`, `services/gemini_document_analysis.py`, `services/firestore_users.py`, `schemas.py` |
| Chat (function tools, unified `POST /v1/chats/{id}/messages`) | `api/routers/chats.py`, `core/chat_orchestrator.py`, `providers/gemini_provider.py`, `persistence/chat_repository.py`, `services/chat_history.py`, `services/gemini_document_analysis.py`, `services/gemini_file_search.py` (File Search store + ingest), `schemas.py` |
| Change model inputs or imputation | `fambot_backend/cardio_features.py`, `services/inference.py`, possibly `model.py` + retrain |
| Retrain or change algorithms | `model.py` |
| Auth behavior | `core/deps.py`, `core/jwt_tokens.py`, `services/identity_toolkit.py`, `core/firebase_init.py` |
| CORS / server bind | `app.py`, env vars |

---

## Style and scope expectations

- Match existing patterns: type hints, `from __future__ import annotations` where already used, thin route handlers delegating to helpers.
- Avoid drive-by refactors unrelated to the requested task.
- Do not add unsolicited README sections or new dependencies without clear need; **do** extend README (and AGENTS when conventions change) when adding features—see [Documentation maintenance](#documentation-maintenance).
- Prefer explicit errors (e.g. `FileNotFoundError` for missing model) over silent fallbacks for production paths.

---

## Testing status

The repository uses **pytest** (`uv sync --group dev`, then `uv run pytest`). Layout:

| Marker / scope | What it covers |
|----------------|----------------|
| Default (`uv run pytest`) | Excludes `external` and `gemini_live` via `addopts = "-m 'not external and not gemini_live'"`. Unit + API tests use mocks, `TestClient`, and/or `FAMBOT_SKIP_*` flags—**no** live Firebase, Storage, Identity Toolkit, or Gemini by default. |
| `unit` | Pure logic (features, JWT helpers, schemas, inference and committed `cardiovascular_model.pkl`, family aggregates, mocked Gemini helpers, holdout ROC-AUC regression on `cardio_train.csv`). |
| `api` | HTTP layer with `TestClient` (health, onboarding with skip flags, JWT auth, mocked auth/storage, documents list/upload/analyze, chats). |
| `external` | Real network or Google APIs. **Also** requires `FAMBOT_RUN_EXTERNAL_TESTS=1` inside tests so `pytest -m external` alone does not surprise-hit production keys without intent. |
| `gemini_live` | Real Gemini `generate_content` smoke in [`tests/external/test_gemini_live.py`](tests/external/test_gemini_live.py). Requires `FAMBOT_RUN_GEMINI_LIVE=1` and `GEMINI_API_KEY`. |

**External / live tests:** `FAMBOT_RUN_EXTERNAL_TESTS=1 uv run pytest -m external` — needs `FIREBASE_WEB_API_KEY` for Identity Toolkit smoke; optional `FAMBOT_EXTERNAL_TEST_EMAIL` / `FAMBOT_EXTERNAL_TEST_PASSWORD` for a successful login check; optional `FAMBOT_EXTERNAL_HEALTH_URL` for a trivial HTTP GET.

**Gemini live smoke:** `FAMBOT_RUN_GEMINI_LIVE=1 uv run pytest -m gemini_live` — requires `GEMINI_API_KEY`; performs a minimal real model call (intended for manual / pre-release checks, not required in CI).

Manual smoke checks still useful after releases:

- `GET /health` without auth
- `PUT /me/onboarding` with `FAMBOT_SKIP_AUTH=1` and `FAMBOT_SKIP_FIRESTORE=1` and a trained model present
- Protected routes with a valid JWT when `FAMBOT_JWT_SECRET` is set and auth is not skipped

---

## Security checklist for changes

- [ ] No default that disables auth in production code paths.
- [ ] No secrets committed (service account JSON, API keys).
- [ ] New user data fields reviewed for PII and Firestore indexing/cost implications.

---

## Versioning

There is **no** URL path version prefix. Release and compatibility are communicated via **`FastAPI(title=…, version=…)`** in `app.py`, OpenAPI metadata, and documentation. Bump the FastAPI `version` string when releasing breaking API changes; document migrations in commit messages or release notes if the project adopts them.

# Agent guide: Fambot Backend

Instructions for AI coding agents and human contributors working in this repository. Read this before making non-trivial changes.

---

## Documentation maintenance

Whenever you add or materially change a **feature** (new or altered endpoints, environment variables, authentication, request/response shapes, Firestore fields, or training/inference contracts), **update the docs in the same effort**:

- **[`README.md`](README.md)** — user-facing overview, env reference, and API reference.
- **This file (`AGENTS.md`)** — when conventions, directory layout, auth flow, or “what to edit for task X” guidance changes.

The repository should remain the **source of truth**; avoid shipping behavior without documenting it.

---

## Project purpose

**Fambot Backend** is a Python service that:

1. Exposes a **FastAPI** HTTP API with **JWT Bearer** authentication (tokens minted by this service after email/password signup/login against Firebase).
2. Persists user onboarding and profile fields in **Google Cloud Firestore** (`users/{uid}`).
3. Loads a **joblib-serialized sklearn Pipeline** (`diabetes_model.pkl`) trained from `model.py` on `sources/diabetes.csv`, and converts onboarding input into a feature row compatible with that pipeline.

Agents should preserve the separation between **training** (`model.py`), **HTTP layer** (`fambot_backend/app.py` and `fambot_backend/api/routers/`), **inference** (`fambot_backend/services/inference.py`), and **persistence** (`fambot_backend/services/firestore_users.py`).

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
| `uv run model` | `[project.scripts] model = "model:main"` | Training pipeline in `model.py`; writes `diabetes_model.pkl` and `feature_importance.png`. |

---

## Directory map

```
fambot_backend/
  app.py                    # FastAPI app, CORS, include_router, uvicorn runner
  schemas.py                # Pydantic models (API JSON shape)
  api/routers/              # health, auth, users (HTTP surface)
  core/
    deps.py                 # HTTPBearer → JWT → firebase_uid (Firebase Auth uid string)
    jwt_tokens.py           # Mint / verify access tokens (FAMBOT_JWT_SECRET)
    firebase_init.py        # firebase_admin.initialize_app (ADC)
  services/
    inference.py            # MODEL_PATH, joblib load, BMI, predict_risk, medians cache
    identity_toolkit.py     # signInWithPassword for POST /auth/login
    firestore_users.py      # get_user_profile, upsert_onboarding, ensure_user_document
model.py                    # Offline training; compares LR vs XGBoost, saves champion
sources/diabetes.csv        # Training data; also used at inference for medians
```

---

## Environment variables agents must respect

| Variable | Agent-relevant behavior |
|----------|-------------------------|
| `MODEL_PATH` | Overrides default `diabetes_model.pkl` path in `services.inference._model_path()`. |
| `FIREBASE_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` | Passed into Firebase app options when set. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Local/service credential path for ADC. |
| `FAMBOT_SKIP_AUTH=1` | **Dev only.** Bypasses JWT verification; uses `FAMBOT_DEV_UID`. Never enable in production code paths or docs that imply production. |
| `FAMBOT_DEV_UID` | UID string when `FAMBOT_SKIP_AUTH=1` (default `dev-user`). |
| `FAMBOT_SKIP_FIRESTORE=1` | Skips Firestore; returns in-memory profile for testing. |
| `FAMBOT_JWT_SECRET` | Required (when auth is not skipped) to sign tokens at signup/login and verify them on protected routes. |
| `FAMBOT_JWT_EXPIRES_SECONDS` | Access token TTL (defaults documented in `core/jwt_tokens.py` / README). |
| `FIREBASE_WEB_API_KEY` | Required for `POST /auth/login` (Identity Toolkit). |
| `FAMBOT_CORS_ORIGINS` | Comma-separated origins; default allows `*`. |

When adding tests or local scripts, prefer `FAMBOT_SKIP_*` flags over mocking unless the test specifically targets Firebase.

---

## API and data conventions

1. **JSON API (Pydantic):** `snake_case` field names (`height_cm`, `blood_pressure_diastolic`, …).
2. **Firestore documents:** `camelCase` keys (`displayName`, `heightCm`, `bloodPressureDiastolic`, …). Mapping lives in `services/firestore_users.py` only—keep it centralized.
3. **Model feature names:** Must match `FEATURE_ORDER` in `services/inference.py` and the column order expected by the saved pipeline. Any new user-facing field that maps into the model requires coordinated updates in:
   - `schemas.OnboardingIn`
   - `services/inference.predict_risk` row construction
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

- `predict_risk` builds a **single-row** `pandas.DataFrame` with columns exactly `FEATURE_ORDER`.
- Missing physiological columns not collected in the API are imputed from **training-set medians** (see `_training_medians()`).
- Risk score is derived from `predict_proba` positive class × 100 when available.
- `compute_bmi` is used for both persistence and the `BMI` feature.

Breaking changes to the saved pipeline (e.g. different column set) require retraining and redeploying `diabetes_model.pkl`.

---

## Training contract (`model.py`)

- Uses stratified train/test split and 5-fold CV.
- Champion is chosen by **CV ROC-AUC** comparison between logistic regression pipeline and XGBoost random search.
- Saves the **entire fitted Pipeline** with `joblib.dump` so inference can call `predict_proba` on the same structure.

If you change preprocessing in `model.py`, mirror any assumptions used at inference (e.g. which columns get zero→NaN) in `services/inference.py` or retrain and ship a new artifact.

---

## Git and generated artifacts

`.gitignore` excludes:

- `.venv`
- `diabetes_model.pkl`
- `feature_importance.png`

The API **fails at runtime** if `diabetes_model.pkl` is missing (unless you point `MODEL_PATH` to a valid file). Agents should mention `uv run model` in docs or setup when adding features that require the model.

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
| Change model inputs or imputation | `services/inference.py`, possibly `model.py` + retrain |
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

There is **no** automated test suite in-repo at the time of this document. Agents adding tests should use `pytest` or the project’s chosen runner once introduced; until then, manual checks:

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

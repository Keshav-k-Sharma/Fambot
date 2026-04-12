# Agent guide: Fambot Backend

Instructions for AI coding agents and human contributors working in this repository. Read this before making non-trivial changes.

---

## Project purpose

**Fambot Backend** is a Python service that:

1. Exposes a **FastAPI** HTTP API with Firebase **Bearer token** authentication.
2. Persists user onboarding and profile fields in **Google Cloud Firestore** (`users/{uid}`).
3. Loads a **joblib-serialized sklearn Pipeline** (`diabetes_model.pkl`) trained from `model.py` on `sources/diabetes.csv`, and converts onboarding input into a feature row compatible with that pipeline.

Agents should preserve the separation between **training** (`model.py`), **HTTP layer** (`fambot_backend/app.py`), **inference** (`fambot_backend/inference.py`), and **persistence** (`fambot_backend/firestore_users.py`).

---

## Tech stack (authoritative)

- **Python** `>= 3.14` per `pyproject.toml`.
- **Package manager:** `uv` is the assumed workflow (`uv sync`, `uv run …`).
- **Web:** FastAPI + Uvicorn.
- **Google:** `firebase-admin` (Auth ID token verification + Firestore).
- **ML:** pandas, scikit-learn, xgboost, joblib, matplotlib (training only).

---

## Entry points and scripts

| Command | Defined in | What it runs |
|---------|------------|----------------|
| `uv run api` | `[project.scripts] api = "fambot_backend.app:run"` | Uvicorn on `fambot_backend.app:app`. |
| `uv run model` | `[project.scripts] model = "model:main"` | Training pipeline in `model.py`; writes `diabetes_model.pkl` and `feature_importance.png`. |

**Not** used as shipped entry points:

- Root `main.py` — placeholder `print`; do not wire new behavior here unless the project explicitly adopts it and `pyproject.toml` is updated.

---

## Directory map

```
fambot_backend/
  app.py              # Routes, CORS, uvicorn runner
  schemas.py          # Pydantic models (API JSON shape)
  inference.py        # MODEL_PATH, joblib load, BMI, predict_risk, medians cache
  deps.py             # HTTPBearer → firebase_uid
  firebase_init.py    # firebase_admin.initialize_app (ADC)
  firestore_users.py  # get_user_profile, upsert_onboarding
model.py              # Offline training; compares LR vs XGBoost, saves champion
sources/diabetes.csv  # Training data; also used at inference for medians
```

---

## Environment variables agents must respect

| Variable | Agent-relevant behavior |
|----------|-------------------------|
| `MODEL_PATH` | Overrides default `diabetes_model.pkl` path in `inference._model_path()`. |
| `FIREBASE_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` | Passed into Firebase app options when set. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Local/service credential path for ADC. |
| `FAMBOT_SKIP_AUTH=1` | **Dev only.** Bypasses token verification; uses `FAMBOT_DEV_UID`. Never enable in production code paths or docs that imply production. |
| `FAMBOT_SKIP_FIRESTORE=1` | Skips Firestore; returns in-memory profile for testing. |
| `FAMBOT_CORS_ORIGINS` | Comma-separated origins; default allows `*`. |

When adding tests or local scripts, prefer `FAMBOT_SKIP_*` flags over mocking unless the test specifically targets Firebase.

---

## API and data conventions

1. **JSON API (Pydantic):** `snake_case` field names (`height_cm`, `blood_pressure_diastolic`, …).
2. **Firestore documents:** `camelCase` keys (`heightCm`, `bloodPressureDiastolic`, …). Mapping lives in `firestore_users.py` only—keep it centralized.
3. **Model feature names:** Must match `FEATURE_ORDER` in `inference.py` and the column order expected by the saved pipeline. Any new user-facing field that maps into the model requires coordinated updates in:
   - `schemas.OnboardingIn`
   - `inference.predict_risk` row construction
   - Optionally `model.py` if training data or preprocessing changes

---

## Authentication flow

1. Client sends `Authorization: Bearer <Firebase ID token>`.
2. `deps.firebase_uid` uses `HTTPBearer` and `firebase_admin.auth.verify_id_token`.
3. If `FAMBOT_SKIP_AUTH=1`, returns a fixed dev UID without calling Firebase.

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

If you change preprocessing in `model.py`, mirror any assumptions used at inference (e.g. which columns get zero→NaN) in `inference.py` or retrain and ship a new artifact.

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
| New endpoint | `app.py`, possibly `schemas.py` |
| Change request validation | `schemas.py` |
| Change Firestore fields | `firestore_users.py`, `schemas.py` (read/write models) |
| Change model inputs or imputation | `inference.py`, possibly `model.py` + retrain |
| Retrain or change algorithms | `model.py` |
| Auth behavior | `deps.py`, `firebase_init.py` |
| CORS / server bind | `app.py`, env vars |

---

## Style and scope expectations

- Match existing patterns: type hints, `from __future__ import annotations` where already used, thin route handlers delegating to helpers.
- Avoid drive-by refactors unrelated to the requested task.
- Do not add unsolicited README sections or new dependencies without clear need.
- Prefer explicit errors (e.g. `FileNotFoundError` for missing model) over silent fallbacks for production paths.

---

## Testing status

There is **no** automated test suite in-repo at the time of this document. Agents adding tests should use `pytest` or the project’s chosen runner once introduced; until then, manual checks:

- `GET /health` without auth
- `PUT /v1/me/onboarding` with `FAMBOT_SKIP_AUTH=1` and `FAMBOT_SKIP_FIRESTORE=1` and a trained model present

---

## Security checklist for changes

- [ ] No default that disables auth in production code paths.
- [ ] No secrets committed (service account JSON, API keys).
- [ ] New user data fields reviewed for PII and Firestore indexing/cost implications.

---

## Versioning

API version is currently reflected in path prefixes (`/v1/…`) and `FastAPI(title=…, version=…)` in `app.py`. Bump the FastAPI `version` string when releasing breaking API changes; document migrations in commit messages or release notes if the project adopts them.

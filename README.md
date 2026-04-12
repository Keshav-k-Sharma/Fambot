# Fambot Backend

HTTP API for user onboarding and **diabetes risk scoring** using a trained scikit-learn / XGBoost pipeline. Clients authenticate with **JWT access tokens** issued by this API after **email/password signup and login**; accounts live in **Firebase Authentication** (server-side Admin SDK + Identity Toolkit), and profiles are stored in **Cloud Firestore**.

This repository is both a **batch training script** (builds `diabetes_model.pkl` from the Pima Indians diabetes dataset) and a **FastAPI** service that serves predictions and persists onboarding data.

**Breaking change:** HTTP routes no longer use a `/v1/` prefix. Clients must call `/auth/…`, `/me/…`, and `/health` directly (for example `POST /auth/login` instead of `POST /v1/auth/login`).

---

## Features

- **Health check** for load balancers and uptime checks.
- **Signup and login** (`POST /auth/signup`, `POST /auth/login`) returning JWT access tokens; Firebase holds the canonical email/password user, Firestore stores profile data by `uid`.
- **Authenticated user profile** (`GET /me`) backed by Firestore document `users/{uid}`.
- **Stored risk score** (`GET /me/risk`) returning the persisted `risk_score` and `risk_class` from the last successful onboarding (no model inference on read).
- **Onboarding completion** (`PUT /me/onboarding`) that:
  - Validates body fields with Pydantic.
  - Computes BMI from height and weight.
  - Runs the ML pipeline to produce a **risk score** (0–100, from the positive-class probability) and **risk class** (`low` / `moderate` / `high`).
  - Merges the result into the user’s Firestore document.

---

## Requirements

- **Python** `>= 3.14` (see [`pyproject.toml`](pyproject.toml)).
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another PEP 517 installer.
- For production-like runs: a **Google Cloud / Firebase** project with:
  - **Application Default Credentials** (ADC), or `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account JSON file.
  - **Firestore** enabled.
  - **Firebase Authentication** (Email/Password sign-in enabled) for signup/login.
  - A **Web API key** for the Identity Toolkit REST API (used server-side for login only; see `FIREBASE_WEB_API_KEY`).
  - A strong **`FAMBOT_JWT_SECRET`** for signing and verifying access tokens.

---

## Repository layout

| Path | Role |
|------|------|
| [`fambot_backend/app.py`](fambot_backend/app.py) | FastAPI app factory, CORS, router includes, `run()` for Uvicorn. |
| [`fambot_backend/api/routers/`](fambot_backend/api/routers/) | HTTP route modules (`health`, `auth`, `me`). |
| [`fambot_backend/schemas.py`](fambot_backend/schemas.py) | Pydantic request/response models. |
| [`fambot_backend/core/deps.py`](fambot_backend/core/deps.py) | JWT Bearer verification → `uid`. |
| [`fambot_backend/core/jwt_tokens.py`](fambot_backend/core/jwt_tokens.py) | Mint and verify HS256 access tokens. |
| [`fambot_backend/core/firebase_init.py`](fambot_backend/core/firebase_init.py) | One-time `firebase_admin` initialization (ADC). |
| [`fambot_backend/services/inference.py`](fambot_backend/services/inference.py) | Model load, BMI, feature row construction, `predict_risk`. |
| [`fambot_backend/services/identity_toolkit.py`](fambot_backend/services/identity_toolkit.py) | Identity Toolkit `signInWithPassword` (login). |
| [`fambot_backend/services/firestore_users.py`](fambot_backend/services/firestore_users.py) | Firestore read/write for `users` collection. |
| [`model.py`](model.py) | Training: logistic regression vs XGBoost, saves `diabetes_model.pkl` and `feature_importance.png`. |
| [`sources/diabetes.csv`](sources/diabetes.csv) | Training data (Pima Indians diabetes CSV). |
| [`diabetes_model.pkl`](diabetes_model.pkl) | Serialized **champion** pipeline (generated; may be gitignored). |

---

## Install

```bash
cd fambot-backend
uv sync
```

Installs the package and dependencies from `pyproject.toml`.

---

## Train the model

Training compares **logistic regression** (with scaling) and **XGBoost** (random search), picks the better **5-fold CV ROC-AUC**, then saves the winning **full sklearn `Pipeline`** with `joblib`.

```bash
uv run model
```

Artifacts (by default next to the project root):

- `diabetes_model.pkl` — required at API runtime unless `MODEL_PATH` overrides the location.
- `feature_importance.png` — bar chart of importances or coefficient magnitudes.

The training script treats zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing in the preprocessing branch used for those columns (Pima-style sentinel zeros).

---

## Run the API

```bash
uv run api
```

This invokes `fambot_backend.app:run`, which starts Uvicorn on:

- **Host:** `HOST` (default `0.0.0.0`)
- **Port:** `PORT` (default `8000`)

Interactive docs (when the server is up):

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HOST` | Bind address for Uvicorn (default `0.0.0.0`). |
| `PORT` | Listen port (default `8000`). |
| `MODEL_PATH` | Optional absolute or relative path to `diabetes_model.pkl`. Defaults to repo-root `diabetes_model.pkl`. |
| `FIREBASE_PROJECT_ID` | Firebase/GCP project ID passed into `firebase_admin.initialize_app`. |
| `GOOGLE_CLOUD_PROJECT` | Alternative to `FIREBASE_PROJECT_ID` for the same purpose. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON for ADC (typical for local dev). |
| `FAMBOT_CORS_ORIGINS` | Comma-separated list of allowed origins for CORS. Default `*` (single origin string `*` in the list). Strip whitespace around entries. |
| `FAMBOT_SKIP_AUTH` | Set to `1` to **skip JWT verification** (local only; **never** in production). When set, the resolved user id comes from `FAMBOT_DEV_UID`. |
| `FAMBOT_DEV_UID` | Fake Firebase `uid` used when `FAMBOT_SKIP_AUTH=1` (default `dev-user`). |
| `FAMBOT_SKIP_FIRESTORE` | Set to `1` to skip Firestore reads/writes (returns synthetic profile data for onboarding). |
| `FAMBOT_JWT_SECRET` | Secret for signing and verifying JWT access tokens (required when auth is not skipped). |
| `FAMBOT_JWT_EXPIRES_SECONDS` | Access token lifetime in seconds (default `3600`; minimum `60`). |
| `FIREBASE_WEB_API_KEY` | Firebase **Web API key** (Identity Toolkit) for `POST /auth/login` only; not a substitute for ADC. |
| `MPLBACKEND` | Used by `model.py` for matplotlib (default `Agg` if unset). |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | **Render only:** entire service account JSON as a string secret; [`scripts/render_start.sh`](scripts/render_start.sh) writes it to a temp file and sets `GOOGLE_APPLICATION_CREDENTIALS`. Not used locally unless you mimic that flow. |

---

## Deployment (Render + Firebase)

Deploy the API as a **Render Web Service** (Python). The repo includes [`render.yaml`](render.yaml) (Blueprint) and [`scripts/render_start.sh`](scripts/render_start.sh) so the service account is provided at runtime without committing JSON keys.

### 1. Firebase / Google Cloud (one-time)

1. In the [Firebase Console](https://console.firebase.google.com/), **create a project** and note the **Project ID** (use the same value for `FIREBASE_PROJECT_ID` on Render).
2. **Authentication** → Sign-in method → enable **Email/Password**.
3. **Firestore** → create a database (choose a location; Native mode is typical).
4. **Project settings** (gear) → **Your apps** → if you need the Web API key, use the **Web API key** from the Firebase config (or Google Cloud Console → APIs & Services → Credentials). Set it as `FIREBASE_WEB_API_KEY` on Render (used only for `POST /auth/login` via Identity Toolkit).
5. **Project settings** → **Service accounts** → **Generate new private key** (JSON). Keep this file private. Paste its **full JSON** into Render as `GOOGLE_SERVICE_ACCOUNT_JSON` (see below). The default Firebase Admin service account can create Auth users and access Firestore when those products are enabled.

Do **not** set `FAMBOT_SKIP_AUTH` or `FAMBOT_SKIP_FIRESTORE` in production.

### 2. Render service

- **Connect** this Git repository to Render and **apply** the Blueprint from [`render.yaml`](render.yaml), or create a **Web Service** manually with the same settings.
- **Build command:** `uv sync && uv run model` (installs deps and generates `diabetes_model.pkl` in the slug; it is gitignored but required at runtime).
- **Start command:** `bash scripts/render_start.sh` (writes `GOOGLE_SERVICE_ACCOUNT_JSON` to `/tmp/gcp-sa.json`, sets `GOOGLE_APPLICATION_CREDENTIALS`, runs `uv run api`).
- **Health check path:** `/health`.
- **Environment:** set `PYTHON_VERSION` to `3.14` if not inherited from [`.python-version`](.python-version). Set the secrets marked `sync: false` in [`render.yaml`](render.yaml) in the dashboard when prompted.

| Render env | Purpose |
|------------|---------|
| `PYTHON_VERSION` | e.g. `3.14` (matches [`pyproject.toml`](pyproject.toml)). |
| `FIREBASE_PROJECT_ID` | Firebase / GCP project ID. |
| `FAMBOT_JWT_SECRET` | Long random secret (e.g. `openssl rand -hex 32`). |
| `FIREBASE_WEB_API_KEY` | Firebase Web API key for Identity Toolkit. |
| `FAMBOT_CORS_ORIGINS` | Comma-separated allowed origins for your web app (avoid `*` in production if you rely on credentialed CORS). |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full JSON from the service account key (single secret). |

Render’s **free** web tier may **spin down** when idle (cold starts). Use a paid instance if you need the service always reachable.

### 3. Smoke test after deploy

Replace the host with your `*.onrender.com` URL:

1. `GET /health` → `{"status":"ok"}`.
2. `POST /auth/signup` then `POST /auth/login` with a test user.
3. `PUT /me/onboarding` with `Authorization: Bearer <JWT>` and confirm data in Firestore `users/{uid}`.

---

## API reference

### `GET /health`

No authentication. Returns `{"status": "ok"}`.

### `POST /auth/signup`

No authentication. **JSON body:** `email`, `password` (minimum length 6 per Firebase), **`name`** (display name, required, max 128 characters after trimming; stored as Firebase Auth `displayName` and on the Firestore user doc as `displayName`).

Creates a Firebase Auth user via the Admin SDK, ensures a minimal Firestore `users/{uid}` document when Firestore is enabled, and returns a **JWT access token** plus `uid`, `email`, `expires_in`, and `token_type`.

**Errors:** `409` if the email is already registered; `400` for invalid Firebase user creation; `500` if `FAMBOT_JWT_SECRET` is not set.

### `POST /auth/login`

No authentication. **JSON body:** `email`, `password`.

Verifies credentials with the Identity Toolkit API (`FIREBASE_WEB_API_KEY` required) and returns the same token shape as signup.

**Errors:** `401` for invalid credentials; `403` if the account is disabled; `500` if `FAMBOT_JWT_SECRET` or `FIREBASE_WEB_API_KEY` is missing; `502` if Identity Toolkit returns a server error (upstream unavailable).

### `GET /me`

**Auth:** `Authorization: Bearer <JWT access token>`

Returns the current user’s profile from Firestore, or an empty-ish profile if the document does not exist. Includes `display_name` when set at signup (`displayName` in Firestore).

**Errors:** `401` if the `Authorization` header is missing or the JWT is invalid/expired; `500` if `FAMBOT_JWT_SECRET` is not set (server cannot verify tokens).

### `GET /me/risk`

**Auth:** Same as `GET /me` (`Authorization: Bearer <JWT access token>`).

Returns the **stored** diabetes risk from Firestore (`riskScore` / `riskClass` written when onboarding completed). Does not run the ML model.

**Response** (`RiskOut`): `risk_score` (0–100) and `risk_class` (`low` \| `moderate` \| `high`).

**Errors:** Same authentication errors as `GET /me`. **`404`** if onboarding is not complete or stored risk fields are missing (e.g. new user or incomplete document).

### `PUT /me/onboarding`

**Auth:** Same as above (`Bearer` JWT).

**Errors:** Same authentication errors as `GET /me`.

**JSON body** (`OnboardingIn`):

| Field | Type | Notes |
|-------|------|--------|
| `age` | int | 1–120 |
| `height_cm` | float | 50–260 |
| `weight_kg` | float | 20–400 |
| `glucose` | float | 1–600 |
| `blood_pressure_diastolic` | float | 20–200; mapped to model feature **BloodPressure** (Pima column is diastolic-like). |
| `blood_pressure_systolic` | float, optional | 40–300; stored in Firestore only. |

**Response** (`OnboardingOut`): includes updated `profile`, `risk_score` (0–100), and `risk_class`.

**Risk buckets** (in [`fambot_backend/services/inference.py`](fambot_backend/services/inference.py)):

- `low`: score below 34  
- `moderate`: score from 34 up to (but not including) 67  
- `high`: score 67 or above  

---

## Firestore schema

Collection: **`users`**, document ID: **Firebase `uid`**.

Fields (camelCase in Firestore):

| Firestore field | Meaning |
|-----------------|--------|
| `displayName` | Display name from signup |
| `age` | User age |
| `heightCm` | Height (cm) |
| `weightKg` | Weight (kg) |
| `glucose` | Plasma glucose |
| `bloodPressureDiastolic` | Diastolic BP |
| `bloodPressureSystolic` | Systolic BP (optional) |
| `bmi` | Computed BMI |
| `riskScore` | 0–100 |
| `riskClass` | `"low"` \| `"moderate"` \| `"high"` |
| `onboardingComplete` | `true` after successful PUT |
| `updatedAt` | Server timestamp (UTC) |

Reads map these back into **snake_case** JSON in `UserProfileOut`.

---

## Inference and the ML pipeline

- The API builds a **single-row DataFrame** with columns in this order:  
  `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- **Pregnancies** is always `0` in the current onboarding flow (no pregnancy field in the API).
- **BloodPressure** uses **diastolic** from the request.
- **SkinThickness**, **Insulin**, and **DiabetesPedigreeFunction** are filled from **medians of the training CSV** (with zeros→NaN for the first two before median), cached in process.
- **BMI** is computed from `height_cm` and `weight_kg`, not taken from the user as a raw field.

The loaded object must be a sklearn estimator with `predict_proba` when possible; the positive class probability is scaled to 0–100.

---

## Security notes

- Production deployments must **not** set `FAMBOT_SKIP_AUTH=1`.
- Keep **`FAMBOT_JWT_SECRET`** long, random, and private; rotate by invalidating old tokens (short `FAMBOT_JWT_EXPIRES_SECONDS` helps).
- Use **HTTPS** in front of the API; protect service account keys and rotate them.
- Firestore **security rules** must restrict `users/{uid}` so clients can only access their own document if clients talk to Firestore directly; this API uses the **Admin SDK** server-side, so rules do not apply to the backend process—protect the backend credentials instead.

---

## Medical and legal disclaimer

The model is trained on **historical tabular data** for research and engineering demonstration. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Any product use requires appropriate clinical and legal review.

---

## License / project metadata

See `pyproject.toml` for package name and version. Add a license file if you distribute this code.

# Fambot Backend

HTTP API for user onboarding and **cardiovascular (heart/circulatory) risk scoring** using a trained scikit-learn / XGBoost (or HistGradientBoosting) pipeline. It is **not** a diabetes product; training data and field names follow a public **cardio** exam survey (ordinal tiers, not lab glucose in mg/dL). Clients authenticate with **JWT access tokens** issued by this API after **email/password signup and login**; accounts live in **Firebase Authentication** (server-side Admin SDK + Identity Toolkit), and profiles are stored in **Cloud Firestore**.

This repository is both a **batch training script** (builds `cardiovascular_model.pkl` from [`sources/cardio_train.csv`](sources/cardio_train.csv)) and a **FastAPI** service that serves predictions and persists onboarding data.

**Breaking change:** HTTP routes no longer use a `/v1/` prefix. Clients must call `/auth/…`, `/me/…`, and `/health` directly (for example `POST /auth/login` instead of `POST /v1/auth/login`).

**API 0.7.0:** Unified chat turn endpoint at `POST /v1/chats/{chat_id}/messages` with `Accept` negotiation (`application/json` buffered or `text/event-stream` SSE). Legacy `POST /chat/{chat_id}` and `POST /chat/{chat_id}/stream` remain as deprecated compatibility routes. **API 0.6.0:** Documents are only under `/documents` (removed `/me/documents/*` and `POST /documents/upload`); upload is `POST /documents` with optional form field `analyze=true`. **API 0.5.0:** Family invitations (`/me/family/…`) with single-use expiring links, QR PNG (base64), and perspective-aware relationship labels. **API 0.4.0** onboarding: `PUT /me/onboarding` uses cardiovascular-aligned fields (gender, BP, cholesterol and **`gluc_ordinal`** survey tiers, optional lifestyle flags). See [Onboarding](#put-meonboarding) below.

---

## Features

- **Health check** for load balancers and uptime checks.
- **Signup and login** (`POST /auth/signup`, `POST /auth/login`) returning JWT access tokens; Firebase holds the canonical email/password user, Firestore stores profile data by `uid`.
- **Authenticated user profile** (`GET /me`) backed by Firestore document `users/{uid}`.
- **Stored risk score** (`GET /me/risk`) returning the persisted `risk_score` and `risk_class` from the last successful onboarding (no model inference on read).
- **Onboarding completion** (`PUT /me/onboarding`) that:
  - Validates body fields with Pydantic.
  - Computes BMI from height and weight.
  - Builds a feature row matching the trained pipeline (including derived BMI, pulse pressure, MAP proxy).
  - Adds optional **family-group features** when the user belongs to a group: relationship-weighted aggregates of **other members’** stored risk scores (from their completed onboarding only). First-degree roles (`mother`, `father`, `son`, `daughter`, `brother`, `sister`, `husband`, `wife`) weight **1.0**; `uncle`, `aunt`, `nephew`, `niece` weight **0.5**; if no relationship edge exists for a peer, weight **0.65**. If no group or no peer scores, those columns are missing and **median-imputed** like omitted lifestyle flags.
  - Runs the ML pipeline to produce a **risk score** (0–100, from the positive-class probability) and **risk class** (`low` / `moderate` / `high`).
  - Merges the result into the user’s Firestore document.
- **Chat sessions + history** (`POST /chat/new`, `GET /chats`, `POST /v1/chats/{chat_id}/messages`, `GET /chat/{chat_id}/history`) with Firestore-backed threads. Chat behavior stays the same (prompting, tool calls, citations, optional attachments), but transport is unified: send `Accept: application/json` for buffered responses or `Accept: text/event-stream` for SSE. SSE now emits structured events (`message_start`, `token`, `tool_call`, `tool_result`, `citation`, `message_end`, `error`) with `chatId`, `turnId`, `sequence`, and `timestamp`. Legacy `/chat/{chat_id}` and `/chat/{chat_id}/stream` routes remain deprecated shims during migration.
- **Documents** (`GET /documents`, `POST /documents`, `GET /documents/{doc_id}`, `GET /documents/{doc_id}/download`, `POST /documents/{doc_id}/analyze`, `DELETE /documents/{doc_id}`): upload/list/get/download/delete medical reports in Firebase Storage under `documents/{uid}/...`; optional **Gemini** analysis on upload (`multipart` form `analyze=true`) or on a stored file by id.
- **Family group (v1)** (`/me/family/…`): group owner creates **single-use** invite links (24h TTL by default) with optional deep-link base URL; response includes **QR code** as base64 PNG. Invitees (existing accounts) **accept** while authenticated; reciprocal relationship labels use a fixed vocabulary and gender-aware mapping. Owner can **remove** members. Each user belongs to **at most one** family group.

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
| [`fambot_backend/cardio_features.py`](fambot_backend/cardio_features.py) | Shared `FEATURE_ORDER` (core vitals + family aggregates), gender mapping, `build_feature_frame` for inference (must match training). |
| [`fambot_backend/api/routers/`](fambot_backend/api/routers/) | HTTP route modules (`health`, `auth`, `me`, `invitations` → `/me/family`). |
| [`fambot_backend/schemas.py`](fambot_backend/schemas.py) | Pydantic request/response models. |
| [`fambot_backend/core/deps.py`](fambot_backend/core/deps.py) | JWT Bearer verification → `uid`. |
| [`fambot_backend/core/jwt_tokens.py`](fambot_backend/core/jwt_tokens.py) | Mint and verify HS256 access tokens. |
| [`fambot_backend/core/firebase_init.py`](fambot_backend/core/firebase_init.py) | One-time `firebase_admin` initialization (ADC). |
| [`fambot_backend/services/inference.py`](fambot_backend/services/inference.py) | `MODEL_PATH`, joblib load, `predict_risk`. |
| [`fambot_backend/services/identity_toolkit.py`](fambot_backend/services/identity_toolkit.py) | Identity Toolkit `signInWithPassword` (login). |
| [`fambot_backend/services/firestore_users.py`](fambot_backend/services/firestore_users.py) | Firestore read/write for `users` collection. |
| [`model.py`](model.py) | Training: LR vs XGB vs HistGradientBoosting; saves `cardiovascular_model.pkl`, `cardiovascular_model.threshold.json`, `feature_importance.png`. |
| [`sources/cardio_train.csv`](sources/cardio_train.csv) | Training data (semicolon-separated). |
| [`cardiovascular_model.pkl`](cardiovascular_model.pkl) | Serialized **champion** pipeline (commit after `uv run model` so Render and other deploys can load it without training at build time). |

---

## Install

```bash
cd fambot-backend
uv sync
cp .env.example .env
# Edit `.env`: set Firebase project ID, Web API key, path to `firebase-admin.json`, and `FAMBOT_JWT_SECRET`.
```

Installs the package and dependencies from `pyproject.toml`. At runtime, [`fambot_backend/app.py`](fambot_backend/app.py) loads variables from a **`.env`** file in the project root (via `python-dotenv`) if present.

---

## Tests

Install dev tools (pytest, httpx) and run the default suite (fast, no live cloud calls):

```bash
uv sync --group dev
uv run pytest
```

Default configuration excludes tests marked `external`. To run live Identity Toolkit / optional HTTP checks:

```bash
FAMBOT_RUN_EXTERNAL_TESTS=1 uv run pytest -m external
```

See [`AGENTS.md`](AGENTS.md) for markers, required env vars (`FIREBASE_WEB_API_KEY`, optional `FAMBOT_EXTERNAL_TEST_EMAIL` / `FAMBOT_EXTERNAL_TEST_PASSWORD`, `FAMBOT_EXTERNAL_HEALTH_URL`), and manual smoke checks.

---

## Train the model

Training compares **logistic regression**, **XGBoost** (random search), and **HistGradientBoostingClassifier** (random search), picks the best **5-fold CV ROC-AUC**, then saves the winning **full sklearn `Pipeline`** with `joblib`.

```bash
uv run model
```

Artifacts (by default next to the project root):

- `cardiovascular_model.pkl` — required at API runtime unless `MODEL_PATH` overrides the location.
- `cardiovascular_model.threshold.json` — optional training metadata (e.g. OOF threshold); not required for serving the 0–100 score.
- `feature_importance.png` — bar chart of importances or coefficient magnitudes.

Training injects **synthetic** family aggregate columns (the public `cardio_train` CSV has no family graph); the live API fills the same columns from the user’s family group when available.

---

## Tests

```bash
uv sync --extra dev
uv run pytest
```

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
| `MODEL_PATH` | Optional absolute or relative path to `cardiovascular_model.pkl`. Defaults to repo-root `cardiovascular_model.pkl`. |
| `FIREBASE_PROJECT_ID` | Firebase/GCP project ID passed into `firebase_admin.initialize_app`. |
| `GOOGLE_CLOUD_PROJECT` | Alternative to `FIREBASE_PROJECT_ID` for the same purpose. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON for ADC (typical for local dev). |
| `FAMBOT_CORS_ORIGINS` | Comma-separated list of allowed origins for CORS. Default `*` (single origin string `*` in the list). Strip whitespace around entries. |
| `FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket name used for report uploads (for example `my-project.appspot.com`). |
| `GEMINI_API_KEY` | API key for Gemini; required for document analysis (`POST /documents` with `analyze=true`, or `POST /documents/{doc_id}/analyze`) and **chat** (including optional File Search indexing on document upload). |
| `GEMINI_REPORT_MODEL` | Optional Gemini model name (default `gemini-2.5-flash`). |
| `FAMBOT_GEMINI_DISABLE_FILE_SEARCH` | Set to `1` to **disable** Gemini **File Search** store creation and upload indexing. Chat turns use custom function tools instead of attaching built-in File Search because Gemini rejects combining those tool types in one request. |
| `FAMBOT_STORAGE_MAKE_PUBLIC` | Optional `1` to make uploaded Storage objects public after upload (default private). |
| `FAMBOT_FAMILY_INVITE_TTL_SECONDS` | Family invite token lifetime (default `86400` = 24h; clamped between 60s and 30 days). |
| `FAMBOT_INVITE_BASE_URL` | Optional URL prefix for invite links and QR payloads (e.g. `https://app.example.com/join`). If unset, invite URLs use the `fambot://family-invite?token=…` scheme. |
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
- **Build command:** `uv sync` (installs dependencies only). **Do not** run `uv run model` on Render unless you want long builds and on-the-fly training. Instead, run `uv run model` locally (or in CI), then **commit** `cardiovascular_model.pkl` at the repo root so it is included in the deploy slug. The API loads it at runtime (see `MODEL_PATH`).
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

Returns the **stored** cardiovascular risk from Firestore (`riskScore` / `riskClass` written when onboarding completed). Does not run the ML model.

**Response** (`RiskOut`): `risk_score` (0–100) and `risk_class` (`low` \| `moderate` \| `high`).

**Errors:** Same authentication errors as `GET /me`. **`404`** if onboarding is not complete or stored risk fields are missing (e.g. new user or incomplete document).

### `PUT /me/onboarding`

**Auth:** Same as above (`Bearer` JWT).

**Errors:** Same authentication errors as `GET /me`. **`422`** if validation fails (e.g. systolic BP not greater than diastolic).

**JSON body** (`OnboardingIn`):

| Field | Type | Notes |
|-------|------|--------|
| `age` | int | 1–120 (years) |
| `height_cm` | float | 120–220 (cm) |
| `weight_kg` | float | 35–250 (kg) |
| `blood_pressure_systolic` | float | 80–250 (mm Hg) |
| `blood_pressure_diastolic` | float | 40–150 (mm Hg); must be **less than** systolic |
| `gender` | `"female"` \| `"male"` | Maps to dataset codes 1 / 2 |
| `cholesterol` | 1, 2, or 3 | Ordinal: 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc_ordinal` | 1, 2, or 3 | Self-reported tier matching training column `gluc` (not a blood test value) |
| `smokes` | bool or omit | Omit to let the model impute from training distribution |
| `drinks_alcohol` | bool or omit | Omit to impute |
| `physically_active` | bool or omit | Omit to impute |

**Response** (`OnboardingOut`): includes updated `profile`, `risk_score` (0–100), and `risk_class`.

Family-aware inputs are **not** part of the JSON body; they are read from the authenticated user’s family group and peer profiles at scoring time (see Features above).

**Risk buckets** (in [`fambot_backend/services/inference.py`](fambot_backend/services/inference.py)):

- `low`: score below 34  
- `moderate`: score from 34 up to (but not including) 67  
- `high`: score 67 or above  

---

### Documents (`/documents`)

All paths require `Authorization: Bearer <JWT>`. Document ids are the stored **filename** under `documents/{uid}/`.

#### `GET /documents`

Lists the authenticated user’s files from Storage.

**Response:** JSON array of `DocumentItem` objects: `id`, `filename`, `content_type`, `size_bytes`, `storage_path`, `storage_uri`, `updated_at`, optional `type`, and `analysis_model` / `recommendations_text` (null when not from an analyzed upload).

#### `POST /documents`

**Content type:** `multipart/form-data`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | file | yes | Medical report or document. |
| `type` | string | no | Optional classification (`DocumentType` enum; same values as before). |
| `analyze` | bool | no | Default `false`. If `true`, saves the file then runs Gemini with Firestore profile context; `analysis_model` and `recommendations_text` are set on the response. |

**Response** (`DocumentItem`): upload metadata; when `analyze=true`, includes `analysis_model` and `recommendations_text`.

**Errors:** `400` empty upload; `401` auth; `500` missing `FIREBASE_STORAGE_BUCKET`; `500`/`502` when `analyze=true` and Gemini is misconfigured or fails.

#### `GET /documents/{doc_id}`

Returns one `DocumentItem` for a user-owned object.

#### `GET /documents/{doc_id}/download`

Streams the **raw file bytes** for viewing or saving (same filename as stored). Response headers include `Content-Type` from storage and `Content-Disposition: attachment` with the original filename (ASCII-safe + RFC 5987 `filename*` for Unicode names).

**Errors:** `404` if the object was deleted from Storage; `401` auth.

#### `POST /documents/{doc_id}/analyze`

Runs Gemini analysis for an **already stored** document.

**Response** (`DocumentAnalysisResult`): `doc_id`, `analysis_model`, `recommendations_text`.

#### `DELETE /documents/{doc_id}`

Deletes the object from Firebase Storage. **Response:** `{"status":"success","detail":"Document deleted"}`.

---

### `POST /chat/new`

**Auth:** `Authorization: Bearer <JWT>`.

No parameters are required.

Creates a Firestore chat session at `users/{uid}/chats/{chat_id}` where `chat_id` is always a server-generated UUID and initial title is always `New chat`.

### `GET /chats`

**Auth:** `Authorization: Bearer <JWT>`.

Lists chat sessions for the authenticated user ordered by `last_updated` descending.

### `POST /v1/chats/{chat_id}/messages`

**Auth:** `Authorization: Bearer <JWT>`.

**Content type:** `multipart/form-data`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `message` | string | yes | User prompt text. |
| `file` | file | no | Optional attachment for this turn. |
| `Idempotency-Key` | header | no | Optional key to dedupe retried turn submissions. |

Sends **user profile + cardiovascular risk** and recent chat history on every turn (no automatic bulk load of all stored documents). The model may use **function tools** (list stored document names, include a file by name, family risk context). Built-in Gemini tools such as `file_search` and `google_search` are not attached to chat turns because Gemini rejects combining built-in tools with custom function declarations in one request. The optional `file` field is for a **this-turn** attachment only. Persists user+model turns under `users/{uid}/chats/{chat_id}/messages`, updates `last_updated`, and tracks turn lifecycle state (`queued`, `streaming`, `completed`, `failed`, `cancelled`) under `users/{uid}/chats/{chat_id}/turns`. After the first user message only, chat title is auto-generated with a low-cost title model.

**Response negotiation (`Accept` header):**

- `Accept: application/json` (default): returns `ChatMessageResponse` with `chat_id`, `turn_id`, `content`, optional `citations`, optional `new_title`, `state`.
- `Accept: text/event-stream`: returns SSE.

### `POST /v1/chats/{chat_id}/messages` (SSE mode)

**Auth:** `Authorization: Bearer <JWT>`.

**Content type:** `multipart/form-data` with `Accept: text/event-stream`.

**Response:** `text/event-stream` (Server-Sent Events). Each line is a `data:` line whose payload is a JSON object:

| `type` | Shape |
|--------|--------|
| `message_start` | `{"type":"message_start","chatId","turnId","sequence","timestamp"}` |
| `token` | `{"type":"token","chatId","turnId","sequence","timestamp","text":"<chunk>"}` |
| `tool_call` | `{"type":"tool_call","chatId","turnId","sequence","timestamp","name","args","tool_call_id"}` |
| `tool_result` | `{"type":"tool_result","chatId","turnId","sequence","timestamp","name","result","tool_call_id"}` |
| `citation` | `{"type":"citation","chatId","turnId","sequence","timestamp","citation":{...}}` |
| `message_end` | `{"type":"message_end","chatId","turnId","sequence","timestamp","state","new_title","citations"}` |
| `error` | `{"type":"error","chatId","turnId","sequence","timestamp","detail":"<message>"}` |

**Client note:** the browser’s native `EventSource` only supports **GET** requests. For this **POST** endpoint, use `fetch` with a streaming body reader (or an equivalent) and parse `data: …` lines. Do not use `EventSource` for this route.

**Errors:** `404` if the chat id does not exist. If generation fails before any bytes are sent, a normal JSON error body may be returned; otherwise an `error` SSE event is emitted.

### Deprecated compatibility routes

- `POST /chat/{chat_id}` — compatibility shim that returns the legacy `ChatInteractionResponse`.
- `POST /chat/{chat_id}/stream` — compatibility shim that emits legacy `text`/`done`/`error` SSE events.

### `GET /chat/{chat_id}/history`

**Auth:** `Authorization: Bearer <JWT>`.

Returns chronological messages for the chat with `role`, `content`, `created_at`, optional `citations`, and optional `has_file`.

---

### `POST /me/family/invitations`

**Auth:** `Authorization: Bearer <JWT>`.

**JSON body** (`CreateFamilyInviteIn`):

| Field | Type | Notes |
|-------|------|--------|
| `target_role` | string | How you describe the person you are inviting (fixed vocabulary: `mother`, `father`, `son`, `daughter`, `brother`, `sister`, `uncle`, `aunt`, `nephew`, `niece`, `husband`, `wife`). |

Only the **group owner** can create invites. The first successful create also creates the family group and adds you as owner.

**Response** (`FamilyInviteCreatedOut`): `token`, `invite_url`, `expires_at` (UTC), `qr_png_base64` (PNG bytes, base64-encoded), `qr_media_type`, `target_role`.

**Errors:** `403` if you are not the owner of your family group; `401` if not authenticated.

### `POST /me/family/invitations/accept`

**Auth:** `Authorization: Bearer <JWT>` (invitees must already have an account and be logged in).

**JSON body** (`AcceptFamilyInviteIn`):

| Field | Type | Notes |
|-------|------|--------|
| `token` | string | Secret token from `POST /me/family/invitations` or parsed from `invite_url`. |

Consumes the invite (single-use), joins the owner’s group, and creates reciprocal relationship edges. If you already belong to another group, you are **removed** from the old group and its relationship edges for you are deleted.

**Response** (`AcceptFamilyInviteOut`): `group_id`, `family` (`FamilyGroupOut`).

**Errors:** `404` invite not found; `410` expired; `409` already used, already in this group, or you **own** a family group (owners cannot join another group in v1); `400` cannot accept your own invite; `401` if not authenticated.

### `GET /me/family`

**Auth:** `Authorization: Bearer <JWT>`.

Returns your current family group, owner id, and **other** members with `role_relative_to_me` when a directed relationship edge exists (may be `null` for pairs not yet modeled).

**Errors:** `404` if you are not in any group; `401` if not authenticated.

### `DELETE /me/family/members/{member_uid}`

**Auth:** `Authorization: Bearer <JWT>`.

**Owner only:** removes a member from the group (cannot remove yourself or the owner). Deletes that member’s `users/{uid}` `familyGroupId` pointer and their relationship edges in the group.

**Response** (`RemoveFamilyMemberOut`): `removed_uid`, `group_id`.

**Errors:** `403` if not owner or if trying to remove the owner; `404` if not in a group or member missing; `401` if not authenticated.

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
| `gender` | `"female"` or `"male"` |
| `cholesterol` | 1, 2, or 3 |
| `glucOrdinal` | 1, 2, or 3 (legacy `glucoseLevel` still read if present) |
| `smokes` | Boolean or null |
| `drinksAlcohol` | Boolean or null |
| `physicallyActive` | Boolean or null |
| `bloodPressureSystolic` | Systolic BP (mm Hg) |
| `bloodPressureDiastolic` | Diastolic BP (mm Hg) |
| `bmi` | Computed BMI |
| `riskScore` | 0–100 |
| `riskClass` | `"low"` \| `"moderate"` \| `"high"` |
| `onboardingComplete` | `true` after successful PUT |
| `updatedAt` | Server timestamp (UTC) |
| `familyGroupId` | Optional string; id of the `familyGroups` document this user belongs to |

Reads map these back into **snake_case** JSON in `UserProfileOut`.

**Family data (additional collections):**

| Collection | Purpose |
|------------|---------|
| `familyGroups/{groupId}` | `ownerUid`, `createdAt`, `updatedAt` |
| `familyGroups/{groupId}/members/{uid}` | `joinedAt` |
| `familyGroups/{groupId}/relationships/{fromUid}__{toUid}` | `fromUid`, `toUid`, `role`, `createdAt` |
| `familyInvites/{token}` | `groupId`, `ownerUid`, `targetRole`, `expiresAt`, `consumedAt`, `consumedByUid` |

---

## Inference and the ML pipeline

- Feature names and order are defined once in [`fambot_backend/cardio_features.py`](fambot_backend/cardio_features.py) (`FEATURE_ORDER`) and must match the columns saved in `cardiovascular_model.pkl`.
- The API builds a **single-row DataFrame** with: `age_years`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc` (from JSON `gluc_ordinal`), `smoke`, `alco`, `active`, plus derived `bmi`, `pulse_pressure`, `map_approx`.
- Optional lifestyle fields omitted from JSON are passed as **missing values** and imputed by the **fitted `SimpleImputer`** inside the saved pipeline (trained on `cardio_train.csv`).
- **BMI** is computed from `height_cm` and `weight_kg` for both persistence and the feature row.

The loaded object must be a sklearn estimator with `predict_proba` when possible; the positive class probability is scaled to 0–100.

---

## Security notes

- Production deployments must **not** set `FAMBOT_SKIP_AUTH=1`.
- Keep **`FAMBOT_JWT_SECRET`** long, random, and private; rotate by invalidating old tokens (short `FAMBOT_JWT_EXPIRES_SECONDS` helps).
- Use **HTTPS** in front of the API; protect service account keys and rotate them.
- Firestore **security rules** must restrict `users/{uid}` so clients can only access their own document if clients talk to Firestore directly; this API uses the **Admin SDK** server-side, so rules do not apply to the backend process—protect the backend credentials instead.

---

## Medical and legal disclaimer

The model is trained on **historical tabular data** for research and engineering demonstration. It does **not** diagnose diabetes or any specific disease; it is **not** a substitute for professional medical advice, diagnosis, or treatment. Any product use requires appropriate clinical and legal review.

---

## License / project metadata

See `pyproject.toml` for package name and version. Add a license file if you distribute this code.

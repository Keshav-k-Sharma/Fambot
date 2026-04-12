#!/usr/bin/env bash
set -euo pipefail
# Render: write service account JSON from a secret env var, then start the API.
# Set GOOGLE_SERVICE_ACCOUNT_JSON in the Render dashboard (full JSON, one line or multiline).
if [[ -z "${GOOGLE_SERVICE_ACCOUNT_JSON:-}" ]]; then
  echo "GOOGLE_SERVICE_ACCOUNT_JSON is not set" >&2
  exit 1
fi
printf '%s' "$GOOGLE_SERVICE_ACCOUNT_JSON" > /tmp/gcp-sa.json
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa.json
exec uv run api

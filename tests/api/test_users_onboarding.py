"""Authenticated /me routes with skip flags (no Firebase)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fambot_backend.cardio_features import Gender
from fambot_backend.services.family_risk_aggregate import neutral_family_features


@pytest.mark.api
def test_put_onboarding_returns_profile_and_risk(
    client: TestClient,
    dry_api_env: None,
    clear_model_cache: None,
) -> None:
    body = {
        "age": 42,
        "height_cm": 172,
        "weight_kg": 73,
        "blood_pressure_systolic": 125,
        "blood_pressure_diastolic": 82,
        "gender": "female",
        "cholesterol_mg_dl": 220,
        "glucose_mg_dl": 110,
        "smokes": False,
        "drinks_alcohol": None,
        "physically_active": True,
    }
    r = client.put("/me/onboarding", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "risk_score" in data and "risk_class" in data
    assert data["profile"]["uid"]
    assert data["profile"]["onboarding_complete"] is True
    assert data["profile"]["gender"] == Gender.female.value
    assert data["profile"]["cholesterol_mg_dl"] == 220
    assert data["profile"]["glucose_mg_dl"] == 110


@pytest.mark.api
def test_put_onboarding_invalid_bp(client: TestClient, dry_api_env: None) -> None:
    r = client.put(
        "/me/onboarding",
        json={
            "age": 40,
            "height_cm": 170,
            "weight_kg": 70,
            "blood_pressure_systolic": 100,
            "blood_pressure_diastolic": 110,
            "gender": "male",
            "cholesterol_mg_dl": 180,
            "glucose_mg_dl": 90,
        },
    )
    assert r.status_code == 422


@pytest.mark.api
def test_put_onboarding_risk_reflects_family_features(
    client: TestClient,
    dry_api_env: None,
    clear_model_cache: None,
) -> None:
    body = {
        "age": 42,
        "height_cm": 172,
        "weight_kg": 73,
        "blood_pressure_systolic": 125,
        "blood_pressure_diastolic": 82,
        "gender": "female",
        "cholesterol_mg_dl": 220,
        "glucose_mg_dl": 110,
        "smokes": False,
        "drinks_alcohol": None,
        "physically_active": True,
    }

    seen_uids: list[str] = []

    def capture_neutral(uid: str) -> dict[str, float | None]:
        seen_uids.append(uid)
        return neutral_family_features()

    def high_family(uid: str) -> dict[str, float | None]:
        seen_uids.append(uid)
        return {
            "fam_weighted_mean_risk": 95.0,
            "fam_max_member_risk": 95.0,
            "fam_first_deg_mean_risk": 95.0,
            "fam_any_member_high_risk": 1.0,
        }

    with patch(
        "fambot_backend.services.inference.compute_family_risk_feature_row",
        side_effect=capture_neutral,
    ):
        r1 = client.put("/me/onboarding", json=body)
    assert r1.status_code == 200, r1.text
    score_neutral = r1.json()["risk_score"]
    assert seen_uids == ["dev-user"]

    seen_uids.clear()
    with patch(
        "fambot_backend.services.inference.compute_family_risk_feature_row",
        side_effect=high_family,
    ):
        r2 = client.put("/me/onboarding", json=body)
    assert r2.status_code == 200, r2.text
    score_high = r2.json()["risk_score"]
    assert seen_uids == ["dev-user"]

    assert abs(score_neutral - score_high) > 0.01

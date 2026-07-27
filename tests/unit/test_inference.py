"""Tests for inference helpers and predict_risk using the committed model artifact."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fambot_backend.cardio_features import Gender
from fambot_backend.schemas import OnboardingIn
from fambot_backend.services.family_risk_aggregate import neutral_family_features
from fambot_backend.services.inference import compute_bmi, predict_risk


@pytest.mark.unit
def test_compute_bmi() -> None:
    bmi = compute_bmi(height_cm=180, weight_kg=81)
    assert abs(bmi - 25.0) < 0.01


@pytest.mark.unit
def test_predict_risk_uses_committed_model(
    clear_model_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    payload = OnboardingIn(
        age=50,
        height_cm=175,
        weight_kg=78,
        blood_pressure_systolic=140,
        blood_pressure_diastolic=90,
        gender=Gender.male,
        cholesterol_mg_dl=220,
        glucose_mg_dl=110,
        smokes=False,
        drinks_alcohol=False,
        physically_active=True,
    )
    score, rclass = predict_risk(payload)
    assert 0.0 <= score <= 100.0
    assert rclass in ("low", "moderate", "high")


def _sample_onboarding() -> OnboardingIn:
    return OnboardingIn(
        age=50,
        height_cm=175,
        weight_kg=78,
        blood_pressure_systolic=140,
        blood_pressure_diastolic=90,
        gender=Gender.male,
        cholesterol_mg_dl=220,
        glucose_mg_dl=110,
        smokes=False,
        drinks_alcohol=False,
        physically_active=True,
    )


@pytest.mark.unit
def test_predict_risk_without_subject_uid_does_not_call_family_row(
    clear_model_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    mock_fam = MagicMock()
    monkeypatch.setattr(
        "fambot_backend.services.inference.compute_family_risk_feature_row",
        mock_fam,
    )
    payload = _sample_onboarding()
    predict_risk(payload)
    mock_fam.assert_not_called()


@pytest.mark.unit
def test_predict_risk_family_features_change_score(
    clear_model_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When subject_uid is set, family aggregates from compute_family_risk_feature_row feed the pipeline."""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    payload = _sample_onboarding()

    def neutral(_uid: str) -> dict[str, float | None]:
        return neutral_family_features()

    def high_family(_uid: str) -> dict[str, float | None]:
        return {
            "fam_weighted_mean_risk": 95.0,
            "fam_max_member_risk": 95.0,
            "fam_first_deg_mean_risk": 95.0,
            "fam_any_member_high_risk": 1.0,
        }

    with patch("fambot_backend.services.inference.compute_family_risk_feature_row", neutral):
        score_neutral, _ = predict_risk(payload, subject_uid="user-1")
    with patch("fambot_backend.services.inference.compute_family_risk_feature_row", high_family):
        score_high, _ = predict_risk(payload, subject_uid="user-1")

    assert abs(score_neutral - score_high) > 0.01

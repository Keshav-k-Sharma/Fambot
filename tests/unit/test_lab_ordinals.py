"""Unit tests for mg/dL → cardio_train ordinal mapping."""

from __future__ import annotations

import pytest

from fambot_backend.cardio_features import (
    cholesterol_mg_dl_to_ordinal,
    glucose_mg_dl_to_ordinal,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mg_dl", "expected"),
    [
        (199.9, 1),
        (200.0, 2),
        (239.9, 2),
        (240.0, 3),
        (300.0, 3),
        (150.0, 1),
    ],
)
def test_cholesterol_mg_dl_to_ordinal(mg_dl: float, expected: int) -> None:
    assert cholesterol_mg_dl_to_ordinal(mg_dl) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mg_dl", "expected"),
    [
        (99.9, 1),
        (100.0, 2),
        (125.9, 2),
        (126.0, 3),
        (200.0, 3),
        (85.0, 1),
    ],
)
def test_glucose_mg_dl_to_ordinal(mg_dl: float, expected: int) -> None:
    assert glucose_mg_dl_to_ordinal(mg_dl) == expected

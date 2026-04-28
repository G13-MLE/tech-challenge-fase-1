"""Testes para src.api.drift.

Testa detecção de data drift com baseline mockada.
"""

from __future__ import annotations

import pytest

from src.api.drift import DriftReport, detect_drift


@pytest.fixture
def mock_reference() -> dict[str, dict[str, object]]:
    """Baseline de referência mockada para testes."""
    return {
        "features": {
            "tenure": {
                "type": "numeric",
                "min": 0.0,
                "max": 72.0,
                "bins": [
                    {"lower": 0.0, "upper": 10.0, "proportion": 0.3},
                    {"lower": 10.0, "upper": 20.0, "proportion": 0.2},
                    {"lower": 20.0, "upper": 72.0, "proportion": 0.5},
                ],
            },
            "MonthlyCharges": {
                "type": "numeric",
                "min": 18.0,
                "max": 120.0,
                "bins": [
                    {"lower": 18.0, "upper": 50.0, "proportion": 0.4},
                    {"lower": 50.0, "upper": 120.0, "proportion": 0.6},
                ],
            },
            "Contract": {
                "type": "categorical",
                "categories": [
                    {"category": "Month-to-month", "proportion": 0.55},
                    {"category": "One year", "proportion": 0.20},
                    {"category": "Two year", "proportion": 0.25},
                ],
            },
        }
    }


def test_detect_drift_no_drift(
    mock_reference: dict[str, dict[str, object]],
) -> None:
    """Valores típicos não devem disparar drift."""
    report = detect_drift(
        {
            "tenure": 15,
            "MonthlyCharges": 60.0,
            "Contract": "Month-to-month",
        },
        reference=mock_reference,
    )

    assert isinstance(report, DriftReport)
    assert report.drift_detected is False
    assert report.drift_score == 0.0
    assert "tenure" in report.features
    assert report.features["tenure"]["score"] == 0.0


def test_detect_drift_numeric_out_of_range(
    mock_reference: dict[str, dict[str, object]],
) -> None:
    """Valor numérico fora do range da baseline deve disparar drift."""
    report = detect_drift(
        {"tenure": 999, "MonthlyCharges": 60.0, "Contract": "Month-to-month"},
        reference=mock_reference,
    )

    assert report.drift_detected is True
    assert report.features["tenure"]["score"] == 1.0


def test_detect_drift_rare_numeric_bin(
    mock_reference: dict[str, dict[str, object]],
) -> None:
    """Bin com proporção muito baixa deve gerar score > 0."""
    # Altera baseline para ter um bin muito raro
    ref = mock_reference.copy()
    ref["features"] = dict(ref["features"])
    ref["features"]["tenure"] = {
        "type": "numeric",
        "min": 0.0,
        "max": 72.0,
        "bins": [
            {"lower": 0.0, "upper": 72.0, "proportion": 0.01},
        ],
    }

    report = detect_drift(
        {"tenure": 5, "MonthlyCharges": 60.0, "Contract": "Month-to-month"},
        reference=ref,
    )

    assert report.drift_detected is True
    assert report.features["tenure"]["score"] > 0.0


def test_detect_drift_unknown_category(
    mock_reference: dict[str, dict[str, object]],
) -> None:
    """Categoria nunca vista no treino deve disparar drift máximo."""
    report = detect_drift(
        {
            "tenure": 15,
            "MonthlyCharges": 60.0,
            "Contract": "Lifetime",
        },
        reference=mock_reference,
    )

    assert report.drift_detected is True
    assert report.features["Contract"]["score"] == 1.0


def test_detect_drift_rare_category(
    mock_reference: dict[str, dict[str, object]],
) -> None:
    """Categoria rara (proporção < 5%) deve gerar drift."""
    ref = mock_reference.copy()
    ref["features"] = dict(ref["features"])
    ref["features"]["Contract"] = {
        "type": "categorical",
        "categories": [
            {"category": "Month-to-month", "proportion": 0.94},
            {"category": "One year", "proportion": 0.02},
            {"category": "Two year", "proportion": 0.04},
        ],
    }

    report = detect_drift(
        {"tenure": 15, "MonthlyCharges": 60.0, "Contract": "One year"},
        reference=ref,
    )

    assert report.drift_detected is True
    assert report.features["Contract"]["score"] > 0.0


def test_detect_drift_uses_default_reference() -> None:
    """detect_drift carrega reference_stats.json quando reference=None."""
    report = detect_drift(
        {
            "tenure": 32,
            "MonthlyCharges": 65.0,
            "Contract": "Month-to-month",
        }
    )

    assert isinstance(report, DriftReport)
    assert "tenure" in report.features
    assert "MonthlyCharges" in report.features
    assert "Contract" in report.features

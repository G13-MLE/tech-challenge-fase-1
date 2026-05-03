"""Testes para src.api.drift e src.api.drift_monitor.

Testa deteccao de data drift per-request com baseline mockada
e calculo de PSI em janela.
"""

from __future__ import annotations

import pytest

from src.api.drift import DriftReport, detect_drift
from src.api.drift_monitor import PsiResult, PsiWindow

_INITIAL_WINDOW_SIZE = 3
_PSI_WINDOW_ADD_COUNT = 3
_MIN_BUF_SIZE = 2


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
    """Valor dentro do range nao dispara drift, mesmo em bin raro."""
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

    assert report.drift_detected is False
    assert report.features["tenure"]["score"] == 0.0


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
    """Categoria existente na baseline, mesmo rara, nao gera drift."""
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

    assert report.drift_detected is False
    assert report.features["Contract"]["score"] == 0.0


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


# ----- Testes para PsiWindow e PsiResult -----

_BASELINE_BINS_EVEN = [
    {"lower": 0.0, "upper": 10.0, "proportion": 0.5},
    {"lower": 10.0, "upper": 20.0, "proportion": 0.5},
]

_BASELINE_BINS_SKEWED = [
    {"lower": 0.0, "upper": 10.0, "proportion": 0.9},
    {"lower": 10.0, "upper": 20.0, "proportion": 0.1},
]


def test_psi_window_initial_state() -> None:
    """Janela comeca vazia e nao pronta."""
    window = PsiWindow.new("tenure", max_size=10)
    assert window.size == 0
    assert window.ready is False


def test_psi_window_add_and_size() -> None:
    """Adicionar valores aumenta tamanho do buffer."""
    window = PsiWindow.new("tenure", max_size=5)
    for i in range(_PSI_WINDOW_ADD_COUNT):
        window.add(float(i))
    assert window.size == _PSI_WINDOW_ADD_COUNT
    assert window.ready is False


def test_psi_window_ready() -> None:
    """Janela fica pronta quando atinge max_size."""
    window = PsiWindow.new("tenure", max_size=_INITIAL_WINDOW_SIZE)
    for i in range(_INITIAL_WINDOW_SIZE):
        window.add(float(i))
    assert window.ready is True


def test_psi_window_overflow() -> None:
    """Buffer circular descarta valores mais antigos alem do max_size."""
    window = PsiWindow.new("tenure", max_size=_MIN_BUF_SIZE)
    window.add(1.0)
    window.add(2.0)
    window.add(3.0)
    assert window.size == _MIN_BUF_SIZE
    assert list(window.buffer) == [2.0, 3.0]


def test_psi_window_reset() -> None:
    """Reset limpa o buffer."""
    window = PsiWindow.new("tenure", max_size=5)
    window.add(1.0)
    window.add(2.0)
    window.reset()
    assert window.size == 0
    assert window.ready is False


def test_psi_identical_distribution() -> None:
    """Distribuicao identica a baseline deve ter PSI ~ 0.0."""
    window = PsiWindow.new("tenure", max_size=4)
    # Metade no primeiro bin, metade no segundo (igual a baseline)
    window.add(5.0)
    window.add(5.0)
    window.add(15.0)
    window.add(15.0)

    psi = window.compute_psi(_BASELINE_BINS_EVEN)
    assert psi == 0.0


def test_psi_different_distribution() -> None:
    """Distribuicao diferente da baseline deve ter PSI > 0.0."""
    window = PsiWindow.new("tenure", max_size=4)
    # Todos no primeiro bin (baseline esperava 50%)
    window.add(5.0)
    window.add(5.0)
    window.add(5.0)
    window.add(5.0)

    psi = window.compute_psi(_BASELINE_BINS_EVEN)
    assert psi > 0.0


def test_psi_empty_window() -> None:
    """Janela vazia retorna PSI 0.0."""
    window = PsiWindow.new("tenure")
    psi = window.compute_psi(_BASELINE_BINS_EVEN)
    assert psi == 0.0


def test_psi_result_status_stable() -> None:
    """PSI < 0.1 deve ser status 'stable'."""
    stable_score = 0.05
    result = PsiResult(feature="tenure", score=stable_score, status="stable")
    assert result.status == "stable"
    assert result.score == stable_score


def test_psi_result_status_moderate() -> None:
    """PSI entre 0.1 e 0.25 deve ser status 'moderate'."""
    result = PsiResult(feature="tenure", score=0.15, status="moderate")
    assert result.status == "moderate"


def test_psi_result_status_significant() -> None:
    """PSI >= 0.25 deve ser status 'significant'."""
    result = PsiResult(feature="tenure", score=0.30, status="significant")
    assert result.status == "significant"


def test_psi_result_from_window_stable() -> None:
    """PsiResult.from_window com distribuicao identica retorna stable."""
    window = PsiWindow.new("tenure", max_size=4)
    window.add(5.0)
    window.add(5.0)
    window.add(15.0)
    window.add(15.0)

    result = PsiResult.from_window(window, _BASELINE_BINS_EVEN)
    assert result.status == "stable"
    assert result.score == 0.0


def test_psi_result_from_window_moderate() -> None:
    """PsiResult.from_window com deslocamento parcial retorna moderate."""
    window = PsiWindow.new("tenure", max_size=4)
    # 3 no primeiro bin (75%), apenas 1 no segundo (25%)
    # vs baseline: 50/50
    window.add(5.0)
    window.add(5.0)
    window.add(5.0)
    window.add(15.0)

    result = PsiResult.from_window(window, _BASELINE_BINS_EVEN)
    assert result.score > 0.0

"""Detecção de data drift em runtime.

Compara features de entrada contra uma baseline de treinamento
(reference_stats.json) usando:
- PSI (Population Stability Index) para features numéricas
- Proporção esperada para features categóricas

Tudo implementado manualmente — sem dependências pesadas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REFERENCE_STATS_PATH = Path(__file__).with_name("reference_stats.json")

# Thresholds de drift (PSI)
_PSI_STABLE = 0.1
_PSI_MODERATE = 0.25

# Threshold para categoria rara (proporção esperada mínima)
_MIN_CATEGORY_PROPORTION = 0.05


def _load_reference_stats(
    path: Path = _REFERENCE_STATS_PATH,
) -> dict[str, Any]:
    """Carrega estatísticas de referência do dataset de treino."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de referência não encontrado: {path}\n"
            "Gere-o com: uv run python -m src.tools.generate_reference_stats"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class DriftReport:
    """Relatório de drift para uma requisição de predição."""

    drift_detected: bool
    """True se alguma feature apresentou drift."""

    drift_score: float
    """Score agregado de drift (soma dos scores por feature)."""

    features: dict[str, dict[str, Any]]
    """Detalhes por feature: score, threshold, tipo."""


def _compute_psi(value: float, baseline: dict[str, Any]) -> float:
    """Calcula PSI para uma feature numérica contra bins de referência.

    Retorna 0.0 se o valor cair em um bin com proporção esperada > 0.
    Se cair em bin vazio ou fora do range, retorna um valor alto.
    """
    bins = baseline["bins"]

    # Encontra em qual bin o valor cai
    for b in bins:
        if b["lower"] <= value <= b["upper"]:
            expected = b["proportion"]
            if expected <= 0:
                return 1.0  # Bin vazio na baseline = certeza de drift
            # PSI simplificado para 1 amostra:
            # (1.0 - expected) * ln(1.0 / expected) quando temos 1 obs
            # Mas como estamos comparando 1 amostra vs baseline:
            # se expected for pequeno (< 0.1), consideramos drift significativo
            if expected < _PSI_STABLE:
                return (_PSI_STABLE - expected) / _PSI_STABLE
            return 0.0

    # Valor fora dos bins da baseline
    return 1.0


def _compute_categorical_drift(value: str, baseline: dict[str, Any]) -> float:
    """Calcula drift para feature categórica.

    Retorna 1.0 se a categoria não existe na baseline ou for rara.
    """
    categories = baseline["categories"]

    for cat in categories:
        if cat["category"] == value:
            proportion = cat["proportion"]
            if proportion < _MIN_CATEGORY_PROPORTION:
                return (
                    _MIN_CATEGORY_PROPORTION - proportion
                ) / _MIN_CATEGORY_PROPORTION
            return 0.0

    # Categoria nunca vista no treino
    return 1.0


def detect_drift(
    features: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> DriftReport:
    """Detecta drift comparando features de entrada contra baseline.

    Args:
        features: Dict com as features da requisição.
            Ex: {"tenure": 5, "MonthlyCharges": 29.85,
            "Contract": "Month-to-month"}
        reference: Stats de referência (carrega do arquivo se None).

    Returns:
        DriftReport com resultado da análise.
    """
    if reference is None:
        reference = _load_reference_stats()["features"]
    else:
        reference = reference.get("features", reference)

    assert reference is not None

    feature_reports: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    drift_detected = False

    for feature_name, baseline in reference.items():
        if feature_name not in features:
            continue

        value = features[feature_name]
        baseline_type = baseline["type"]

        if baseline_type == "numeric":
            score = _compute_psi(float(value), baseline)
        elif baseline_type == "categorical":
            score = _compute_categorical_drift(str(value), baseline)
        else:
            continue

        feature_reports[feature_name] = {
            "score": round(score, 4),
            "type": baseline_type,
            "value": value,
        }

        total_score += score
        if score > 0.0:
            drift_detected = True

    return DriftReport(
        drift_detected=drift_detected,
        drift_score=round(total_score, 4),
        features=feature_reports,
    )

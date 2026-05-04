"""Detecção de data drift em runtime.

Compara features de entrada contra uma baseline de treinamento
(reference_stats.json) usando:

- Range check [min, max] para features numericas (per-request)
- Categorias ineditas para features categoricas (per-request)

Para PSI real por janela de amostras, veja src.api.drift_monitor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REFERENCE_STATS_PATH = Path(__file__).with_name("reference_stats.json")


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


def _compute_range_drift(value: float, baseline: dict[str, Any]) -> float:
    """Calcula score de drift para uma feature numerica.

    Para comparacao single-sample, considera drift apenas se o valor
    estiver fora do range [min, max] do treino. Valores dentro do
    range, mesmo em bins raros, nao sao considerados drift.
    """
    min_val = baseline["min"]
    max_val = baseline["max"]

    if value < min_val or value > max_val:
        return 1.0

    return 0.0


def _compute_categorical_drift(value: str, baseline: dict[str, Any]) -> float:
    """Calcula score de drift para feature categorica.

    Para comparacao single-sample, considera drift apenas se a
    categoria nunca foi vista nos dados de treino.
    """
    categories = baseline["categories"]

    for cat in categories:
        if cat["category"] == value:
            return 0.0

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
        if value is None:
            continue
        baseline_type = baseline["type"]

        if baseline_type == "numeric":
            score = _compute_range_drift(float(value), baseline)
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

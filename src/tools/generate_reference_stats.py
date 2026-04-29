"""Gera reference_stats.json para detecção de data drift.

Calcula distribuições de referência (baseline) a partir do dataset
original de treino. Usado pelo detector de drift em runtime para
comparar features de entrada contra a distribuição histórica.

Uso:
    uv run python -m src.tools.generate_reference_stats
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_DATASET_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
_OUTPUT_PATH = Path("src/api/reference_stats.json")

# Features monitoradas para drift (devem bater com PredictRequest)
_NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
_CATEGORICAL_FEATURES = ["Contract"]

# Número de bins para features numéricas
_NUM_BINS = 10


def _compute_numeric_distribution(
    series: pd.Series, num_bins: int
) -> dict[str, Any]:
    """Calcula histograma com bins fixos para uma feature numérica."""
    min_val = float(series.min())
    max_val = float(series.max())
    counts = pd.cut(series, bins=num_bins, include_lowest=True, retbins=True)[
        0
    ]
    bin_counts = counts.value_counts().sort_index()

    total = len(series)
    bins: list[dict[str, Any]] = []
    for interval, count in bin_counts.items():  # type: ignore[union-attr]
        bins.append(
            {
                "lower": float(interval.left),  # type: ignore[union-attr]
                "upper": float(interval.right),  # type: ignore[union-attr]
                "count": int(count),
                "proportion": round(count / total, 4),
            }
        )

    return {
        "type": "numeric",
        "min": min_val,
        "max": max_val,
        "mean": float(series.mean()),
        "std": float(series.std()),
        "num_bins": num_bins,
        "bins": bins,
    }


def _compute_categorical_distribution(
    series: pd.Series,
) -> dict[str, Any]:
    """Calcula frequências para uma feature categórica."""
    value_counts = series.value_counts()
    total = len(series)

    categories: list[dict[str, Any]] = []
    for category, count in value_counts.items():
        categories.append(
            {
                "category": str(category),
                "count": int(count),
                "proportion": round(count / total, 4),
            }
        )

    return {
        "type": "categorical",
        "num_categories": len(value_counts),
        "categories": categories,
    }


def generate_reference_stats(
    dataset_path: Path = _DATASET_PATH,
    output_path: Path = _OUTPUT_PATH,
) -> dict[str, Any]:
    """Lê o dataset e gera estatísticas de referência para drift."""
    df = pd.read_csv(dataset_path)

    stats: dict[str, Any] = {
        "dataset": str(dataset_path),
        "total_rows": len(df),
        "features": {},
    }

    for feature in _NUMERIC_FEATURES:
        stats["features"][feature] = _compute_numeric_distribution(
            df[feature], _NUM_BINS
        )

    for feature in _CATEGORICAL_FEATURES:
        stats["features"][feature] = _compute_categorical_distribution(
            df[feature]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] Reference stats gerado: {output_path}")
    print(f"     Total de rows: {stats['total_rows']}")
    print(f"     Features: {list(stats['features'].keys())}")

    return stats


def main() -> None:
    generate_reference_stats()


if __name__ == "__main__":
    main()

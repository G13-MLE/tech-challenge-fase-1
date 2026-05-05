"""Pipeline de comparacao entre MLP e modelos baseline.

Este script orquestra o treinamento e avaliacao de todos os
modelos (DummyClassifier, Logistic Regression e MLP), gera
metricas comparativas, analise de trade-off e visualizacoes.

Uso:
    $ uv run python -m src.pipelines.run_compare_models
    $ uv run python -m src.pipelines.run_compare_models \
        --input path/to/data.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.logging import setup_logging
from src.constants import DEFAULT_DATASET_PATH
from src.pipelines.common import load_dotenv_silent, set_global_seed
from src.training.compare_models import (
    ModelResult,
    build_comparison_table,
    build_threshold_comparison,
    generate_markdown_report,
    plot_confusion_matrices,
    plot_cost_comparison,
    plot_metrics_radar,
    plot_pr_comparison,
    plot_roc_comparison,
    plot_threshold_tradeoff,
    prepare_data,
    train_and_evaluate_dummy,
    train_and_evaluate_logistic,
    train_and_evaluate_mlp,
)

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


def _log_model_metrics(
    results: dict[str, ModelResult],
) -> None:
    """Log metricas de cada modelo no resultado."""
    for name, result in results.items():
        logger.info(
            "  %s: accuracy=%.4f f1=%.4f roc_auc=%.4f",
            name,
            result.metrics.get("accuracy", 0.0),
            result.metrics.get("f1_score", 0.0),
            result.metrics.get("roc_auc", 0.0),
        )


def _save_outputs(
    all_results: dict[str, ModelResult],
    comparison_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    reports_dir: Path,
) -> None:
    """Salva CSVs e threshold tradeoffs por modelo."""
    comparison_csv = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("Comparacao salva em: %s", comparison_csv)

    if not threshold_df.empty:
        threshold_csv = reports_dir / "threshold_comparison.csv"
        threshold_df.to_csv(threshold_csv, index=False)
        logger.info("Thresholds salvos em: %s", threshold_csv)

    for name, result in all_results.items():
        if result.threshold_df is not None:
            safe_name = name.lower().replace(" ", "_")
            path = reports_dir / f"threshold_tradeoff_{safe_name}.csv"
            result.threshold_df.to_csv(path, index=False)

    # Relatorio markdown
    logger.info("Gerando relatorio docs/MLP_VERSUS_BASELINE.md...")
    markdown = generate_markdown_report(
        all_results, comparison_df, threshold_df
    )
    report_path = Path("docs/MLP_VERSUS_BASELINE.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    logger.info("Relatorio salvo em: %s", report_path)


def main() -> int:
    """Ponto de entrada do script de comparacao.

    Orquestra:
    1. Carregamento e preparacao dos dados
    2. Treinamento e avaliacao de DummyClassifier
    3. Treinamento e avaliacao de Logistic Regression
    4. Treinamento e avaliacao de MLP
    5. Geracao de tabela comparativa e relatorio
    """
    parser = argparse.ArgumentParser(
        description="Comparacao MLP vs modelos baseline"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_DATASET_PATH,
        help="Caminho para o dataset de entrada",
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Pular treinamento MLP (usar apenas baselines)",
    )
    args = parser.parse_args()

    load_dotenv_silent()
    setup_logging()

    np.random.seed(42)
    set_global_seed(42)

    logger.info("Carregando e preparando dados...")
    data = prepare_data(args.input)
    logger.info(
        "Treino: %d amostras, Teste: %d amostras",
        len(data["y_train"]),
        len(data["y_test"]),
    )

    all_results: dict[str, ModelResult] = {}

    # --- DummyClassifier ---
    logger.info("Treinando DummyClassifier (3 estrategias)...")
    dummy_results = train_and_evaluate_dummy(data)
    all_results.update(dummy_results)
    _log_model_metrics(dummy_results)

    # --- Logistic Regression ---
    logger.info("Treinando Logistic Regression...")
    logistic_results = train_and_evaluate_logistic(data)
    all_results.update(logistic_results)
    _log_model_metrics(logistic_results)

    # --- MLP ---
    if not args.skip_mlp:
        logger.info("Treinando MLP...")
        mlp_results = train_and_evaluate_mlp(data)
        all_results.update(mlp_results)
        _log_model_metrics(mlp_results)
    else:
        logger.info("Treinamento MLP skipped (--skip-mlp)")

    # --- Tabela comparativa ---
    logger.info("Gerando tabela comparativa...")
    comparison_df = build_comparison_table(all_results)
    logger.info("\n%s", comparison_df.to_string(index=False))

    # --- Threshold comparison ---
    threshold_df = build_threshold_comparison(all_results)
    if not threshold_df.empty:
        logger.info(
            "\nThreshold otimo por modelo:\n%s",
            threshold_df.to_string(index=False),
        )

    # --- Visualizacoes ---
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Gerando visualizacoes comparativas...")
    plot_roc_comparison(all_results, reports_dir / "comparison_roc_curve.png")
    plot_pr_comparison(all_results, reports_dir / "comparison_pr_curve.png")
    plot_confusion_matrices(
        all_results, reports_dir / "comparison_confusion_matrices.png"
    )
    plot_cost_comparison(comparison_df, reports_dir / "comparison_cost.png")
    plot_threshold_tradeoff(
        all_results, reports_dir / "comparison_threshold_tradeoff.png"
    )
    plot_metrics_radar(
        all_results, reports_dir / "comparison_metrics_radar.png"
    )

    # --- Salva CSVs e relatorio ---
    _save_outputs(all_results, comparison_df, threshold_df, reports_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Testes para o modulo de comparacao de modelos.

Testa funcoes de build de tabela comparativa, geracao de
threshold comparison e geracao de relatorio markdown, com
dados sinteticos para evitar dependencia de treino real.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import THRESHOLD
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
)

_MIN_MD_LENGTH = 100


@pytest.fixture
def sample_results() -> dict[str, ModelResult]:
    """Cria resultados sinteticos para testes."""
    np.random.seed(42)
    n = 100
    y_true = np.array([0] * 73 + [1] * 27)
    np.random.shuffle(y_true)

    y_proba_dummy = np.full(n, 0.27)
    y_pred_dummy = (y_proba_dummy > THRESHOLD).astype(int)

    y_proba_logistic = np.clip(
        y_true * 0.7 + np.random.normal(0, 0.15, n), 0, 1
    )
    y_pred_logistic = (y_proba_logistic > THRESHOLD).astype(int)

    y_proba_mlp = np.clip(y_true * 0.8 + np.random.normal(0, 0.1, n), 0, 1)
    y_pred_mlp = (y_proba_mlp > THRESHOLD).astype(int)

    threshold_df = pd.DataFrame(
        {
            "threshold": [0.3, 0.4, 0.5],
            "precision": [0.4, 0.6, 0.7],
            "recall": [0.9, 0.7, 0.5],
            "f1_score": [0.55, 0.65, 0.58],
            "accuracy": [0.6, 0.75, 0.8],
            "total_cost": [10000, 8000, 9000],
            "cost_false_negatives": [8000, 5000, 4000],
            "cost_false_positives": [2000, 3000, 5000],
            "false_positives": [40, 60, 100],
            "false_negatives": [16, 10, 8],
        }
    )

    return {
        "DummyClassifier_stratified": ModelResult(
            model_name="DummyClassifier_stratified",
            metrics={
                "accuracy": 0.55,
                "precision": 0.30,
                "recall": 0.50,
                "f1_score": 0.37,
                "roc_auc": 0.52,
                "pr_auc": 0.30,
                "brier_score": 0.25,
            },
            y_true=y_true,
            y_pred=y_pred_dummy,
            y_proba=y_proba_dummy,
            cost_analysis={
                "total_cost": 25000.0,
                "normalized_cost": 250.0,
                "cost_false_negatives": 13500.0,
                "cost_false_positives": 11500.0,
            },
            confusion_matrix_dict={
                "true_negatives": 55,
                "false_positives": 18,
                "false_negatives": 27,
                "true_positives": 0,
            },
            threshold_df=None,
            calibration=None,
        ),
        "LogisticRegression": ModelResult(
            model_name="LogisticRegression",
            metrics={
                "accuracy": 0.80,
                "precision": 0.65,
                "recall": 0.55,
                "f1_score": 0.60,
                "roc_auc": 0.84,
                "pr_auc": 0.65,
                "brier_score": 0.15,
            },
            y_true=y_true,
            y_pred=y_pred_logistic,
            y_proba=y_proba_logistic,
            cost_analysis={
                "total_cost": 12000.0,
                "normalized_cost": 120.0,
                "cost_false_negatives": 6000.0,
                "cost_false_positives": 6000.0,
            },
            confusion_matrix_dict={
                "true_negatives": 65,
                "false_positives": 8,
                "false_negatives": 12,
                "true_positives": 15,
            },
            threshold_df=threshold_df.copy(),
            calibration={
                "brier_score": 0.15,
                "expected_calibration_error": 0.08,
            },
        ),
        "MLP": ModelResult(
            model_name="MLP",
            metrics={
                "accuracy": 0.82,
                "precision": 0.67,
                "recall": 0.60,
                "f1_score": 0.63,
                "roc_auc": 0.86,
                "pr_auc": 0.68,
                "brier_score": 0.13,
            },
            y_true=y_true,
            y_pred=y_pred_mlp,
            y_proba=y_proba_mlp,
            cost_analysis={
                "total_cost": 10500.0,
                "normalized_cost": 105.0,
                "cost_false_negatives": 5000.0,
                "cost_false_positives": 5500.0,
            },
            confusion_matrix_dict={
                "true_negatives": 66,
                "false_positives": 7,
                "false_negatives": 10,
                "true_positives": 17,
            },
            threshold_df=threshold_df.copy(),
            calibration={
                "brier_score": 0.13,
                "expected_calibration_error": 0.06,
            },
        ),
    }


class TestBuildComparisonTable:
    """Testes para build_comparison_table."""

    @staticmethod
    def test_builds_dataframe_with_metrics(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        assert isinstance(table, pd.DataFrame)
        assert "metric" in table.columns
        assert "DummyClassifier_stratified" in table.columns
        assert "LogisticRegression" in table.columns
        assert "MLP" in table.columns

    @staticmethod
    def test_contains_key_metrics(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        metrics = table["metric"].tolist()
        assert "roc_auc" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "total_cost" in metrics

    @staticmethod
    def test_confusion_matrix_in_table(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        metrics = table["metric"].tolist()
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert "true_positives" in metrics


class TestBuildThresholdComparison:
    """Testes para build_threshold_comparison."""

    @staticmethod
    def test_returns_dataframe(
        sample_results: dict[str, ModelResult],
    ) -> None:
        df = build_threshold_comparison(sample_results)
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        assert "optimal_threshold" in df.columns

    @staticmethod
    def test_skips_models_without_threshold(
        sample_results: dict[str, ModelResult],
    ) -> None:
        df = build_threshold_comparison(sample_results)
        model_names = df["model"].tolist()
        assert "LogisticRegression" in model_names
        assert "MLP" in model_names
        assert "DummyClassifier_stratified" not in model_names


class TestGenerateMarkdownReport:
    """Testes para generate_markdown_report."""

    @staticmethod
    def test_generates_string(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        threshold = build_threshold_comparison(sample_results)
        md = generate_markdown_report(sample_results, table, threshold)
        assert isinstance(md, str)
        assert len(md) > _MIN_MD_LENGTH

    @staticmethod
    def test_contains_key_sections(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        threshold = build_threshold_comparison(sample_results)
        md = generate_markdown_report(sample_results, table, threshold)
        assert "Tabela Comparativa" in md
        assert "Trade-off" in md
        assert "Conclusoes" in md
        assert "threshold" in md.lower() or "Threshold" in md
        assert "ROC-AUC" in md or "roc_auc" in md

    @staticmethod
    def test_contains_model_names(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        threshold = build_threshold_comparison(sample_results)
        md = generate_markdown_report(sample_results, table, threshold)
        assert "LogisticRegression" in md
        assert "MLP" in md
        assert "DummyClassifier_stratified" in md


class TestPlots:
    """Testes para funcoes de plot."""

    @staticmethod
    def test_plot_roc_comparison(
        sample_results: dict[str, ModelResult],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roc.png"
            plot_roc_comparison(sample_results, path)
            assert path.exists()

    @staticmethod
    def test_plot_pr_comparison(
        sample_results: dict[str, ModelResult],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pr.png"
            plot_pr_comparison(sample_results, path)
            assert path.exists()

    @staticmethod
    def test_plot_confusion_matrices(
        sample_results: dict[str, ModelResult],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cm.png"
            plot_confusion_matrices(sample_results, path)
            assert path.exists()

    @staticmethod
    def test_plot_cost_comparison(
        sample_results: dict[str, ModelResult],
    ) -> None:
        table = build_comparison_table(sample_results)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cost.png"
            plot_cost_comparison(table, path)
            assert path.exists()

    @staticmethod
    def test_plot_threshold_tradeoff(
        sample_results: dict[str, ModelResult],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "threshold.png"
            plot_threshold_tradeoff(sample_results, path)
            assert path.exists()

    @staticmethod
    def test_plot_metrics_radar(
        sample_results: dict[str, ModelResult],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "radar.png"
            plot_metrics_radar(sample_results, path)
            assert path.exists()

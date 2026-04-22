"""Tests for metrics module (src.training.metrics).

Tests the canonical implementation of compute_binary_classification_metrics
and additional diagnostic functions (confusion matrix, calibration, cost).
"""

# ruff: noqa: PLR2004

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.metrics import (
    analyze_threshold_tradeoff,
    compute_binary_classification_metrics,
    compute_calibration_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_cost_analysis,
)


class TestComputeBinaryClassificationMetrics:
    """Test canonical implementation in src.training.metrics."""

    @staticmethod
    @pytest.mark.fast
    def test_perfect_predictions_numpy() -> None:
        """Should compute correct metrics with numpy arrays."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    @staticmethod
    @pytest.mark.fast
    def test_with_probabilities_numpy() -> None:
        """Should compute AUC metrics with probabilities."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.9])

        metrics = compute_binary_classification_metrics(y_true, y_pred, y_prob)

        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0

    @staticmethod
    @pytest.mark.fast
    def test_zero_division_handling() -> None:
        """Should handle zero division gracefully."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    @staticmethod
    @pytest.mark.fast
    def test_torch_tensor_inputs() -> None:
        """Should handle PyTorch tensors directly."""
        y_true = torch.tensor([1, 0, 1, 1])
        y_pred = torch.tensor([1, 0, 0, 1])
        y_prob = torch.tensor([0.9, 0.2, 0.3, 0.8])

        metrics = compute_binary_classification_metrics(
            y_true, y_pred, y_prob, positive_label=None
        )

        assert isinstance(metrics["accuracy"], float)
        assert "roc_auc" in metrics

    @staticmethod
    @pytest.mark.fast
    def test_pandas_series_with_string_labels() -> None:
        """Should handle pandas Series with string labels."""
        y_true = pd.Series(["Yes", "No", "Yes", "No"])
        y_pred = pd.Series(["Yes", "No", "No", "No"])
        y_prob = pd.Series([0.9, 0.1, 0.3, 0.2])

        metrics = compute_binary_classification_metrics(
            y_true, y_pred, y_prob, positive_label="Yes"
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "roc_auc" in metrics

    @staticmethod
    @pytest.mark.fast
    def test_numpy_arrays_without_positive_label() -> None:
        """Should work with numpy arrays without positive_label."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" not in metrics  # no probabilities

    @staticmethod
    @pytest.mark.fast
    def test_pandas_series_numeric_without_label() -> None:
        """Should work with numeric pandas Series without label."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 0, 0])

        metrics = compute_binary_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_score" in metrics


class TestConfusionMatrix:
    """Test compute_confusion_matrix."""

    @staticmethod
    @pytest.mark.fast
    def test_basic_counts() -> None:
        """Should return correct TN, FP, FN, TP."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        cm = compute_confusion_matrix(y_true, y_pred)

        assert cm["true_negatives"] == 2
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 1
        assert cm["true_positives"] == 2

    @staticmethod
    @pytest.mark.fast
    def test_perfect_predictions() -> None:
        """Should return zero off-diagonal entries."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        cm = compute_confusion_matrix(y_true, y_pred)

        assert cm["false_positives"] == 0
        assert cm["false_negatives"] == 0
        assert cm["true_negatives"] == 2
        assert cm["true_positives"] == 2

    @staticmethod
    @pytest.mark.fast
    def test_with_string_labels() -> None:
        """Should handle string labels with positive_label."""
        y_true = pd.Series(["Yes", "No", "Yes", "No"])
        y_pred = pd.Series(["Yes", "Yes", "No", "No"])

        cm = compute_confusion_matrix(y_true, y_pred, positive_label="Yes")

        assert cm["true_negatives"] == 1
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 1
        assert cm["true_positives"] == 1


class TestClassificationReport:
    """Test compute_classification_report."""

    @staticmethod
    @pytest.mark.fast
    def test_structure() -> None:
        """Should return dict with class keys and metric sub-keys."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        report = compute_classification_report(y_true, y_pred)

        # Deve conter entradas por classe e agregados
        assert "class_0" in report
        assert "class_1" in report
        assert "macro avg" in report
        assert "accuracy" in report

        # Cada classe deve ter precision, recall, f1-score
        for key in ("precision", "recall", "f1-score", "support"):
            assert key in report["class_1"]

    @staticmethod
    @pytest.mark.fast
    def test_perfect_predictions() -> None:
        """Should report 1.0 for all per-class metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        report = compute_classification_report(y_true, y_pred)

        assert report["class_1"]["precision"] == 1.0
        assert report["class_1"]["recall"] == 1.0
        assert report["class_1"]["f1-score"] == 1.0

    @staticmethod
    @pytest.mark.fast
    def test_with_string_labels() -> None:
        """Should use positive_label in target names."""
        y_true = pd.Series(["Yes", "No", "Yes", "No"])
        y_pred = pd.Series(["Yes", "Yes", "No", "No"])

        report = compute_classification_report(
            y_true, y_pred, positive_label="Yes"
        )

        assert "Yes" in report
        assert "not_Yes" in report


class TestCalibrationMetrics:
    """Test compute_calibration_metrics."""

    @staticmethod
    @pytest.mark.fast
    def test_perfect_calibration() -> None:
        """Should return low Brier score for calibrated probs."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])

        cal = compute_calibration_metrics(y_true, y_prob)

        assert "brier_score" in cal
        assert "expected_calibration_error" in cal
        assert cal["brier_score"] == pytest.approx(0.0, abs=1e-6)
        assert cal["expected_calibration_error"] == pytest.approx(
            0.0, abs=1e-6
        )

    @staticmethod
    @pytest.mark.fast
    def test_uncertain_probs() -> None:
        """Should handle uncertain probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        cal = compute_calibration_metrics(y_true, y_prob)

        # Brier score para prob 0.5 constante
        assert cal["brier_score"] == pytest.approx(0.25, abs=1e-6)
        assert cal["expected_calibration_error"] >= 0.0

    @staticmethod
    @pytest.mark.fast
    def test_pandas_input() -> None:
        """Should accept pandas Series."""
        y_true = pd.Series([0, 1, 0, 1])
        y_prob = pd.Series([0.1, 0.9, 0.2, 0.8])

        cal = compute_calibration_metrics(y_true, y_prob)

        assert "brier_score" in cal
        assert isinstance(cal["brier_score"], float)


class TestCostAnalysis:
    """Test compute_cost_analysis."""

    @staticmethod
    @pytest.mark.fast
    def test_symmetric_costs() -> None:
        """Should compute total cost as sum of errors."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        # TN=2, FP=1, FN=1, TP=2

        cost = compute_cost_analysis(y_true, y_pred, cost_fn=1.0, cost_fp=1.0)

        assert cost["total_cost"] == 2.0  # 1*1 + 1*1
        assert cost["cost_false_negatives"] == 1.0
        assert cost["cost_false_positives"] == 1.0
        assert cost["normalized_cost"] == pytest.approx(2.0 / 6.0, abs=1e-6)

    @staticmethod
    @pytest.mark.fast
    def test_asymmetric_costs() -> None:
        """Should weight FN and FP differently."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])  # FN=1, FP=0, TN=2, TP=1

        cost = compute_cost_analysis(y_true, y_pred, cost_fn=5.0, cost_fp=1.0)

        assert cost["total_cost"] == 5.0  # 1*5 + 0*1
        assert cost["cost_false_negatives"] == 5.0
        assert cost["cost_false_positives"] == 0.0

    @staticmethod
    @pytest.mark.fast
    def test_savings_and_wasted() -> None:
        """Should compute savings_vs_no_action and wasted_retention."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        # TP=2, FP=1

        cost = compute_cost_analysis(y_true, y_pred, cost_fn=3.0, cost_fp=2.0)

        assert cost["savings_vs_no_action"] == 6.0  # 2 * 3.0
        assert cost["wasted_retention"] == 2.0  # 1 * 2.0

    @staticmethod
    @pytest.mark.fast
    def test_with_string_labels() -> None:
        """Should compute cost with string labels."""
        y_true = pd.Series(["Yes", "No", "Yes", "No"])
        y_pred = pd.Series(["Yes", "Yes", "No", "No"])

        cost = compute_cost_analysis(
            y_true,
            y_pred,
            cost_fn=10.0,
            cost_fp=2.0,
            positive_label="Yes",
        )

        assert cost["cost_false_negatives"] == pytest.approx(1.0 * 10.0)
        assert cost["cost_false_positives"] == pytest.approx(1.0 * 2.0)
        assert cost["total_cost"] == 12.0


class TestThresholdTradeoff:
    """Test analyze_threshold_tradeoff."""

    @staticmethod
    @pytest.mark.fast
    def test_returns_dataframe() -> None:
        """Should return a DataFrame with expected columns."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])

        df = analyze_threshold_tradeoff(y_true, y_prob)

        assert isinstance(df, pd.DataFrame)
        expected_columns = {
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "total_cost",
            "cost_false_negatives",
            "cost_false_positives",
            "false_positives",
            "false_negatives",
        }
        assert set(df.columns) >= expected_columns
        assert len(df) > 0

    @staticmethod
    @pytest.mark.fast
    def test_threshold_0_5_reasonable() -> None:
        """Should find good metrics near threshold 0.5."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])

        df = analyze_threshold_tradeoff(y_true, y_prob)
        row_05 = df[df["threshold"] == 0.5].iloc[0]

        # Com threshold 0.5, espera-se acurácia alta para este dataset
        assert row_05["accuracy"] >= 0.5
        assert row_05["precision"] >= 0.0
        assert row_05["recall"] >= 0.0

    @staticmethod
    @pytest.mark.fast
    def test_custom_thresholds() -> None:
        """Should accept custom thresholds array."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.4, 0.6])
        thresholds = np.array([0.3, 0.5, 0.7])

        df = analyze_threshold_tradeoff(y_true, y_prob, thresholds)

        assert len(df) == 3
        assert list(df["threshold"].values) == [0.3, 0.5, 0.7]

    @staticmethod
    @pytest.mark.fast
    def test_pandas_input() -> None:
        """Should handle pandas Series input."""
        y_true = pd.Series([0, 1, 0, 1])
        y_prob = pd.Series([0.1, 0.9, 0.2, 0.8])

        df = analyze_threshold_tradeoff(y_true, y_prob)

        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns

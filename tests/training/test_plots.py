"""Tests for src.training.plots module.

Lightweight tests that verify plot files are created without errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.training.plots import (
    save_calibration_curve,
    save_confusion_matrix_plot,
    save_loss_curve,
    save_pr_curve,
    save_roc_curve,
)


class TestPlots:
    """Test plot generators."""

    @staticmethod
    @pytest.mark.fast
    def test_save_pr_curve(tmp_path: Path) -> None:
        """Should create a PR curve PNG file."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
        filepath = tmp_path / "pr_curve.png"

        save_pr_curve(y_true, y_prob, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    @staticmethod
    @pytest.mark.fast
    def test_save_calibration_curve(tmp_path: Path) -> None:
        """Should create a calibration curve PNG file."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
        filepath = tmp_path / "calibration_curve.png"

        save_calibration_curve(y_true, y_prob, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    @staticmethod
    @pytest.mark.fast
    def test_save_roc_curve(tmp_path: Path) -> None:
        """Should create a ROC curve PNG file."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
        filepath = tmp_path / "roc_curve.png"

        save_roc_curve(y_true, y_prob, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    @staticmethod
    @pytest.mark.fast
    def test_save_confusion_matrix_plot(tmp_path: Path) -> None:
        """Should create a confusion matrix PNG file."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        filepath = tmp_path / "confusion_matrix.png"

        save_confusion_matrix_plot(y_true, y_pred, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    @staticmethod
    @pytest.mark.fast
    def test_save_confusion_matrix_plot_custom_labels(
        tmp_path: Path,
    ) -> None:
        """Should create confusion matrix with custom labels."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        filepath = tmp_path / "cm_custom.png"

        save_confusion_matrix_plot(
            y_true, y_pred, str(filepath), labels=["Neg", "Pos"]
        )

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    @staticmethod
    @pytest.mark.fast
    def test_save_loss_curve(tmp_path: Path) -> None:
        """Should create a loss curve PNG file."""
        train_losses = [0.8, 0.6, 0.4, 0.3, 0.25]
        val_losses = [0.9, 0.7, 0.5, 0.45, 0.42]
        filepath = tmp_path / "loss_curve.png"

        save_loss_curve(train_losses, val_losses, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

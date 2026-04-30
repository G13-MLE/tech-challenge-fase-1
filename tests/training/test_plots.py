"""Tests for src.training.plots module.

Lightweight tests that verify plot files are created without errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.training.plots import save_calibration_curve, save_pr_curve


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

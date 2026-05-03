"""Tests for src.inference.recover_model module.

Tests model recovery functions with mocked MLflow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.inference.recover_model import (
    get_latest_run_id,
    recover_dummy_model,
    recover_logistic_model,
    recover_mlp_model,
)


class TestGetLatestRunId:
    """Tests for get_latest_run_id."""

    @staticmethod
    @pytest.mark.fast
    def test_raises_when_experiment_not_found() -> None:
        """Should raise RuntimeError if experiment does not exist."""
        with (
            patch(
                "src.inference.recover_model.mlflow.get_experiment_by_name",
                return_value=None,
            ),
            patch("src.inference.recover_model.load_dotenv_silent"),
            pytest.raises(
                RuntimeError,
                match="nao encontrado",
            ),
        ):
            get_latest_run_id("nonexistent")

    @staticmethod
    @pytest.mark.fast
    def test_raises_when_no_runs() -> None:
        """Should raise RuntimeError if no runs exist."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "123"
        with (
            patch(
                "src.inference.recover_model.mlflow.get_experiment_by_name",
                return_value=mock_exp,
            ),
            patch(
                "src.inference.recover_model.mlflow.search_runs",
                return_value=pd.DataFrame(),
            ),
            patch("src.inference.recover_model.load_dotenv_silent"),
            pytest.raises(
                RuntimeError,
                match="Nenhum run",
            ),
        ):
            get_latest_run_id("empty_exp")

    @staticmethod
    @pytest.mark.fast
    def test_returns_run_id_when_run_exists() -> None:
        """Should return run_id from most recent run."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "123"
        runs_df = pd.DataFrame({"run_id": ["abc456"]})
        with (
            patch(
                "src.inference.recover_model.mlflow.get_experiment_by_name",
                return_value=mock_exp,
            ),
            patch(
                "src.inference.recover_model.mlflow.search_runs",
                return_value=runs_df,
            ),
            patch("src.inference.recover_model.load_dotenv_silent"),
        ):
            result = get_latest_run_id("test_exp")

        assert result == "abc456"


class TestRecoverMlpModel:
    """Tests for recover_mlp_model."""

    @staticmethod
    @pytest.mark.fast
    def test_recover_mlp_model_calls_load_model(
        tmp_path: Path,
    ) -> None:
        """Should call mlflow.pytorch.load_model with correct URI."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}

        with (
            patch(
                "mlflow.pytorch.load_model",
                return_value=mock_model,
            ) as mock_load,
            patch("mlflow.set_tracking_uri"),
            patch("torch.save"),
            patch("src.inference.recover_model.load_dotenv_silent"),
        ):
            recover_mlp_model(run_id="test123", output_dir=str(tmp_path))

        mock_load.assert_called_once_with("runs:/test123/model")

    @staticmethod
    @pytest.mark.fast
    def test_recover_mlp_model_finds_latest_run(
        tmp_path: Path,
    ) -> None:
        """Should find latest run when no run_id provided."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}

        with (
            patch(
                "src.inference.recover_model.get_latest_run_id",
                return_value="latest_run",
            ),
            patch(
                "mlflow.pytorch.load_model",
                return_value=mock_model,
            ),
            patch("mlflow.set_tracking_uri"),
            patch("torch.save"),
            patch("src.inference.recover_model.load_dotenv_silent"),
        ):
            result = recover_mlp_model(output_dir=str(tmp_path))

        assert "mlp_model" in str(result)


class TestRecoverLogisticModel:
    """Tests for recover_logistic_model."""

    @staticmethod
    @pytest.mark.fast
    def test_recover_logistic_model_calls_load_model(
        tmp_path: Path,
    ) -> None:
        """Should call mlflow.sklearn.load_model with correct URI."""
        mock_model = MagicMock()

        with (
            patch(
                "mlflow.sklearn.load_model",
                return_value=mock_model,
            ) as mock_load,
            patch("joblib.dump"),
            patch("mlflow.set_tracking_uri"),
            patch(
                "src.inference.recover_model.get_latest_run_id",
                return_value="log_run",
            ),
            patch("src.inference.recover_model.load_dotenv_silent"),
        ):
            recover_logistic_model(run_id="log_run", output_dir=str(tmp_path))

        mock_load.assert_called_once_with("runs:/log_run/model")


class TestRecoverDummyModel:
    """Tests for recover_dummy_model."""

    @staticmethod
    @pytest.mark.fast
    def test_recover_dummy_model_with_strategy(
        tmp_path: Path,
    ) -> None:
        """Should recover dummy model by strategy."""
        mock_model = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "456"
        runs_df = pd.DataFrame({"run_id": ["dummy_run"]})

        with (
            patch(
                "src.inference.recover_model.mlflow.get_experiment_by_name",
                return_value=mock_exp,
            ),
            patch(
                "src.inference.recover_model.mlflow.search_runs",
                return_value=runs_df,
            ) as mock_search,
            patch(
                "mlflow.sklearn.load_model",
                return_value=mock_model,
            ),
            patch("joblib.dump"),
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("src.inference.recover_model.load_dotenv_silent"),
        ):
            result = recover_dummy_model(
                strategy="most_frequent",
                output_dir=str(tmp_path),
            )

        assert "dummy_most_frequent_model" in str(result)
        mock_search.assert_called_once()
        kwargs = mock_search.call_args.kwargs
        assert "filter_string" in kwargs
        assert (
            kwargs["filter_string"]
            == "tags.`mlflow.runName` = 'dummy_most_frequent'"
        )

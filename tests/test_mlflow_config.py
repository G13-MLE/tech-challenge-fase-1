"""Tests for mlflow_config module."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from src.config.mlflow_config import (
    MLflowConfigError,
    setup_logging,
    setup_mlflow,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    @staticmethod
    def test_setup_logging_default_level():
        """Test setup_logging with default level."""
        setup_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    @staticmethod
    def test_setup_logging_custom_level():
        """Test setup_logging with custom level."""
        setup_logging(logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG


@patch("src.config.mlflow_config.mlflow")
def test_setup_mlflow_defaults(mock_mlflow):
    """Test setup_mlflow with default arguments."""
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()

    with patch.dict(os.environ, {}, clear=True):
        result = setup_mlflow()

    assert result == "tech-challenge-fase-1"
    mock_mlflow.set_tracking_uri.assert_called_once_with("file:./mlruns")
    mock_mlflow.set_experiment.assert_called_once_with("tech-challenge-fase-1")


@patch("src.config.mlflow_config.mlflow")
def test_setup_mlflow_custom_experiment(mock_mlflow):
    """Test setup_mlflow with custom experiment name."""
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()

    result = setup_mlflow(experiment_name="custom-exp")

    assert result == "custom-exp"
    mock_mlflow.set_experiment.assert_called_once_with("custom-exp")


@patch("src.config.mlflow_config.mlflow")
def test_setup_mlflow_custom_tracking_uri(mock_mlflow):
    """Test setup_mlflow with custom tracking URI."""
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()

    result = setup_mlflow(
        experiment_name="test-exp", tracking_uri="http://localhost:5000"
    )

    assert result == "test-exp"
    mock_mlflow.set_tracking_uri.assert_called_once_with(
        "http://localhost:5000"
    )


@patch("src.config.mlflow_config.mlflow")
def test_setup_mlflow_env_tracking_uri(mock_mlflow):
    """Test setup_mlflow reads MLFLOW_TRACKING_URI from environment."""
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()

    with patch.dict(
        os.environ, {"MLFLOW_TRACKING_URI": "http://prod-server:5000"}
    ):
        result = setup_mlflow()

    assert result == "tech-challenge-fase-1"
    mock_mlflow.set_tracking_uri.assert_called_once_with(
        "http://prod-server:5000"
    )


@patch("src.config.mlflow_config.mlflow")
def test_setup_mlflow_uri_precedence(mock_mlflow):
    """Test explicit URI takes precedence over env var."""
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()

    with patch.dict(
        os.environ, {"MLFLOW_TRACKING_URI": "http://env-server:5000"}
    ):
        result = setup_mlflow(tracking_uri="http://explicit-server:5000")

    assert result == "tech-challenge-fase-1"
    mock_mlflow.set_tracking_uri.assert_called_once_with(
        "http://explicit-server:5000"
    )


class TestMLflowConfigError:
    """Tests for MLflowConfigError exception."""

    @staticmethod
    def test_exception_message():
        """Test exception can be raised with message."""
        msg = "Test error message"
        with pytest.raises(MLflowConfigError, match=msg):
            raise MLflowConfigError(msg)

    @staticmethod
    def test_exception_inheritance():
        """Test MLflowConfigError inherits from Exception."""
        assert issubclass(MLflowConfigError, Exception)

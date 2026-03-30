"""Tests for mlflow_config module."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from src.config.mlflow_config import (
    MAX_PORT,
    Environment,
    MLflowConfig,
    MLflowConfigError,
    get_mlflow_port,
    setup_logging,
    setup_mlflow,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test Environment enum has correct values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_environment_count(self):
        """Test Environment has exactly 3 values."""
        assert len(Environment) == 3


class TestMLflowConfigError:
    """Tests for MLflowConfigError exception."""

    def test_exception_message(self):
        """Test exception can be raised with message."""
        msg = "Test error message"
        with pytest.raises(MLflowConfigError, match=msg):
            raise MLflowConfigError(msg)

    def test_exception_inheritance(self):
        """Test MLflowConfigError inherits from Exception."""
        assert issubclass(MLflowConfigError, Exception)


class TestMLflowConfig:
    """Tests for MLflowConfig class."""

    def test_defaults_constant_exists(self):
        """Test _DEFAULTS class attribute exists."""
        assert hasattr(MLflowConfig, "_DEFAULTS")
        assert isinstance(MLflowConfig._DEFAULTS, dict)

    def test_defaults_has_all_environments(self):
        """Test _DEFAULTS has all environments."""
        assert Environment.DEVELOPMENT in MLflowConfig._DEFAULTS
        assert Environment.STAGING in MLflowConfig._DEFAULTS
        assert Environment.PRODUCTION in MLflowConfig._DEFAULTS

    def test_init_default_values(self):
        """Test __init__ with default values."""
        config = MLflowConfig()

        assert config.tracking_uri == "file:./mlruns"
        assert config.experiment_name == "tech-challenge-fase-1"
        assert config.port == 5000
        assert config.host == "127.0.0.1"
        assert config.artifact_root == "./mlruns"
        assert config.environment == Environment.DEVELOPMENT

    def test_init_custom_values(self):
        """Test __init__ with custom values."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test-exp",
            port=6000,
            host="localhost",
            artifact_root="/tmp/mlruns",
        )

        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "test-exp"
        assert config.port == 6000
        assert config.host == "localhost"
        assert config.artifact_root == "/tmp/mlruns"

    def test_init_staging_environment(self):
        """Test __init__ with staging environment."""
        config = MLflowConfig(environment=Environment.STAGING)

        assert config.environment == Environment.STAGING
        assert config.port == 5001
        assert config.host == "0.0.0.0"

    def test_init_production_environment(self):
        """Test __init__ with production environment."""
        with patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": "http://prod-server:5000",
                "MLFLOW_ARTIFACT_ROOT": "s3://bucket/",
                "MLFLOW_PORT": "6000",
                "MLFLOW_HOST": "prod-server",
            },
        ):
            config = MLflowConfig(environment=Environment.PRODUCTION)

            assert config.environment == Environment.PRODUCTION
            assert config.tracking_uri == "http://prod-server:5000"
            assert config.artifact_root == "s3://bucket/"
            assert config.port == 6000
            assert config.host == "prod-server"

    def test_for_env_factory_method(self):
        """Test for_env factory method."""
        config = MLflowConfig.for_env(Environment.STAGING)

        assert config.environment == Environment.STAGING
        assert config.port == 5001

    def test_for_env_with_kwargs(self):
        """Test for_env with additional kwargs."""
        config = MLflowConfig.for_env(
            Environment.DEVELOPMENT,
            experiment_name="custom-exp",
            port=7000,
        )

        assert config.experiment_name == "custom-exp"
        assert config.port == 7000

    def test_from_env_vars(self):
        """Test from_env_vars class method."""
        config = MLflowConfig.from_env_vars()

        assert config.environment == Environment.DEVELOPMENT

    def test_validate_config_missing_tracking_uri(self):
        """Test validation fails with missing tracking_uri in PRODUCTION."""
        # In PRODUCTION, tracking_uri defaults to
        # os.getenv("MLFLOW_TRACKING_URI")
        # which returns None if not set, triggering validation
        with patch.dict(os.environ, {}, clear=True):
            # Clear all MLFLOW_ env vars
            for key in list(os.environ.keys()):
                if key.startswith("MLFLOW_"):
                    del os.environ[key]
            with pytest.raises(
                MLflowConfigError, match="tracking_uri é obrigatório"
            ):
                MLflowConfig(environment=Environment.PRODUCTION)

    def test_validate_config_missing_experiment_name(self):
        """Test validation fails with missing experiment_name."""
        # Pass explicit None to override all defaults
        config = MLflowConfig.__new__(MLflowConfig)
        config.logger = logging.getLogger(__name__)
        config.tracking_uri = "file:./mlruns"
        config.port = 5000
        config.host = "127.0.0.1"
        config.artifact_root = "./mlruns"
        config.environment = Environment.DEVELOPMENT
        config.experiment_name = None  # Explicitly None

        with pytest.raises(
            MLflowConfigError, match="experiment_name é obrigatório"
        ):
            config._validate_config()

    def test_validate_config_invalid_port_low(self):
        """Test validation fails with port too low."""
        # port=-1 will not fall back to default since -1 is truthy for 'or'
        # but it's invalid
        with pytest.raises(MLflowConfigError, match="Porta inválida"):
            MLflowConfig(port=-1)

    def test_validate_config_invalid_port_high(self):
        """Test validation fails with port too high."""
        with pytest.raises(MLflowConfigError, match="Porta inválida"):
            MLflowConfig(port=MAX_PORT + 1)

    def test_validate_config_valid_port_boundary(self):
        """Test validation passes with valid boundary ports."""
        config_low = MLflowConfig(port=1)
        assert config_low.port == 1

        config_high = MLflowConfig(port=MAX_PORT)
        assert config_high.port == MAX_PORT

    def test_validate_production_local_tracking_uri_warning(self, caplog):
        """Test warning for production with local tracking URI."""
        with caplog.at_level(logging.WARNING):
            MLflowConfig(
                environment=Environment.PRODUCTION,
                tracking_uri="file:./mlruns",
            )

        assert "tracking_uri deve usar servidor remoto" in caplog.text

    def test_get_artifact_path_file_uri(self):
        """Test get_artifact_path with file URI."""
        config = MLflowConfig(tracking_uri="file:./mlruns")

        path = config.get_artifact_path()

        assert str(path) == "."

    def test_get_artifact_path_remote_uri(self):
        """Test get_artifact_path with remote URI."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            artifact_root="/custom/path",
        )

        path = config.get_artifact_path()

        assert str(path) == "/custom/path"

    @patch("src.config.mlflow_config.mlflow")
    def test_setup_success(self, mock_mlflow):
        """Test setup method success."""
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        config = MLflowConfig()
        result = config.setup()

        assert result == "tech-challenge-fase-1"
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()

    @patch("src.config.mlflow_config.mlflow")
    def test_setup_mlflow_exception(self, mock_mlflow):
        """Test setup handles MlflowException."""
        from mlflow.exceptions import MlflowException

        mock_mlflow.set_tracking_uri.side_effect = MlflowException("Error")

        config = MLflowConfig()

        with pytest.raises(MLflowConfigError, match="Falha ao configurar"):
            config.setup()

    @patch("src.config.mlflow_config.mlflow")
    def test_setup_unexpected_exception(self, mock_mlflow):
        """Test setup handles unexpected exceptions."""
        mock_mlflow.set_tracking_uri.side_effect = RuntimeError("Unexpected")

        config = MLflowConfig()

        with pytest.raises(
            MLflowConfigError, match="Erro inesperado na configuração"
        ):
            config.setup()


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default_level(self):
        """Test setup_logging with default level."""
        setup_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom level."""
        setup_logging(logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG


class TestSetupMlflow:
    """Tests for setup_mlflow function."""

    @patch("src.config.mlflow_config.mlflow")
    def test_setup_mlflow_default(self, mock_mlflow):
        """Test setup_mlflow with default arguments."""
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        result = setup_mlflow()

        assert result == "tech-challenge-fase-1"

    @patch("src.config.mlflow_config.mlflow")
    def test_setup_mlflow_custom_experiment(self, mock_mlflow):
        """Test setup_mlflow with custom experiment name."""
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        result = setup_mlflow(experiment_name="custom-exp")

        assert result == "custom-exp"

    @patch("src.config.mlflow_config.mlflow")
    def test_setup_mlflow_staging_environment(self, mock_mlflow):
        """Test setup_mlflow with staging environment."""
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()

        result = setup_mlflow(environment=Environment.STAGING)

        assert result == "tech-challenge-fase-1"


class TestGetMlflowPort:
    """Tests for get_mlflow_port function."""

    def test_get_mlflow_port_default(self):
        """Test get_mlflow_port returns default port."""
        with pytest.warns(DeprecationWarning):
            port = get_mlflow_port()

        assert port == 5000

    def test_get_mlflow_port_from_env(self):
        """Test get_mlflow_port reads from environment."""
        with (
            patch.dict(os.environ, {"MLFLOW_PORT": "6000"}),
            pytest.warns(DeprecationWarning),
        ):
            port = get_mlflow_port()

        assert port == 6000

    def test_get_mlflow_port_deprecation_warning(self):
        """Test get_mlflow_port emits deprecation warning."""
        with pytest.warns(DeprecationWarning, match="deprecado"):
            get_mlflow_port()


class TestConstants:
    """Tests for module constants."""

    def test_max_port_value(self):
        """Test MAX_PORT constant has correct value."""
        assert MAX_PORT == 65535

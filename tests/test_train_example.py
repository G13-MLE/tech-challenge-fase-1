"""Tests for train_example module."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config.mlflow_config import Environment, MLflowConfigError
from src.train_example import (
    TrainingError,
    evaluate_model,
    load_data,
    log_to_mlflow,
    train_example,
    train_model,
)


class TestTrainingError:
    """Tests for TrainingError exception."""

    def test_training_error_message(self):
        """Test TrainingError can be raised with message."""
        msg = "Test training error"
        with pytest.raises(TrainingError, match=msg):
            raise TrainingError(msg)

    def test_training_error_inheritance(self):
        """Test TrainingError inherits from Exception."""
        assert issubclass(TrainingError, Exception)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_returns_dict(self):
        """Test load_data returns dictionary."""
        data = load_data()

        assert isinstance(data, dict)
        assert "X_train" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_test" in data

    def test_load_data_shapes(self):
        """Test load_data returns correct shapes."""
        data = load_data(test_size=0.3, random_state=42)

        # Iris has 150 samples
        total_samples = data["X_train"].shape[0] + data["X_test"].shape[0]
        assert total_samples == 150

        # With test_size=0.3, train should have ~105 samples
        assert data["X_train"].shape[0] == 105
        assert data["X_test"].shape[0] == 45

    def test_load_data_custom_params(self):
        """Test load_data with custom parameters."""
        data = load_data(test_size=0.5, random_state=123)

        assert data["X_train"].shape[0] == 75
        assert data["X_test"].shape[0] == 75

    def test_load_data_feature_shapes(self):
        """Test load_data feature dimensions."""
        data = load_data()

        # Iris has 4 features
        assert data["X_train"].shape[1] == 4
        assert data["X_test"].shape[1] == 4

    def test_load_data_reproducibility(self):
        """Test load_data produces same results with same random_state."""
        data1 = load_data(random_state=42)
        data2 = load_data(random_state=42)

        np.testing.assert_array_equal(data1["X_train"], data2["X_train"])
        np.testing.assert_array_equal(data1["y_train"], data2["y_train"])


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        data = load_data()
        return data["X_train"], data["y_train"]

    def test_train_model_returns_model(self, sample_data):
        """Test train_model returns RandomForestClassifier."""
        X_train, y_train = sample_data
        model = train_model(X_train, y_train)

        assert isinstance(model, RandomForestClassifier)

    def test_train_model_default_params(self, sample_data):
        """Test train_model with default parameters."""
        X_train, y_train = sample_data
        model = train_model(X_train, y_train)

        assert model.n_estimators == 100
        assert model.max_depth == 10

    def test_train_model_custom_params(self, sample_data):
        """Test train_model with custom parameters."""
        X_train, y_train = sample_data
        model = train_model(
            X_train,
            y_train,
            params={
                "n_estimators": 50,
                "max_depth": 5,
                "random_state": 123,
            },
        )

        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_train_model_fitted(self, sample_data):
        """Test train_model returns fitted model."""
        X_train, y_train = sample_data
        model = train_model(X_train, y_train)

        # Check model has been fitted by predicting
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)

    def test_train_model_merges_params(self, sample_data):
        """Test train_model merges custom params with defaults."""
        X_train, y_train = sample_data
        model = train_model(
            X_train,
            y_train,
            params={"n_estimators": 200},
        )

        assert model.n_estimators == 200
        # Default max_depth should still be 10
        assert model.max_depth == 10


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create trained model and test data."""
        data = load_data()
        model = train_model(data["X_train"], data["y_train"])
        return model, data["X_test"], data["y_test"]

    def test_evaluate_model_returns_dict(self, trained_model_and_data):
        """Test evaluate_model returns dictionary."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert isinstance(metrics, dict)

    def test_evaluate_model_keys(self, trained_model_and_data):
        """Test evaluate_model returns correct keys."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_evaluate_model_values_range(self, trained_model_and_data):
        """Test evaluate_model metrics are in valid range."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        for metric_name, metric_value in metrics.items():
            assert 0.0 <= metric_value <= 1.0, (
                f"{metric_name} out of range: {metric_value}"
            )

    def test_evaluate_model_high_accuracy(self, trained_model_and_data):
        """Test evaluate_model achieves high accuracy on Iris."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        # Random forest on Iris should achieve > 90% accuracy
        assert metrics["accuracy"] > 0.9


class TestLogToMlflow:
    """Tests for log_to_mlflow function."""

    @pytest.fixture
    def trained_model_and_metrics(self):
        """Create trained model and metrics."""
        data = load_data()
        model = train_model(data["X_train"], data["y_train"])
        metrics = evaluate_model(model, data["X_test"], data["y_test"])
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }
        return model, params, metrics

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_logs_params(
        self, mock_mlflow, trained_model_and_metrics
    ):
        """Test log_to_mlflow logs parameters."""
        model, params, metrics = trained_model_and_metrics

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        log_to_mlflow(model, params, metrics)

        mock_mlflow.log_params.assert_called_once_with(params)

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_logs_metrics(
        self, mock_mlflow, trained_model_and_metrics
    ):
        """Test log_to_mlflow logs metrics."""
        model, params, metrics = trained_model_and_metrics

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        log_to_mlflow(model, params, metrics)

        assert mock_mlflow.log_metric.call_count == len(metrics)

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_logs_model(
        self, mock_mlflow, trained_model_and_metrics
    ):
        """Test log_to_mlflow logs model."""
        model, params, metrics = trained_model_and_metrics

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        log_to_mlflow(model, params, metrics)

        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_with_tags(
        self, mock_mlflow, trained_model_and_metrics
    ):
        """Test log_to_mlflow with tags."""
        model, params, metrics = trained_model_and_metrics
        tags = {"version": "v1.0.0", "author": "test"}

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        log_to_mlflow(model, params, metrics, tags=tags)

        assert mock_mlflow.set_tag.call_count == len(tags)

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_returns_run_id(
        self, mock_mlflow, trained_model_and_metrics
    ):
        """Test log_to_mlflow returns run ID."""
        model, params, metrics = trained_model_and_metrics

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        run_id = log_to_mlflow(model, params, metrics)

        assert run_id == "test-run-id"


class TestTrainExample:
    """Tests for train_example function."""

    @patch("src.train_example.setup_mlflow")
    @patch("src.train_example.mlflow")
    @patch("src.train_example.setup_logging")
    def test_train_example_returns_metrics(
        self, mock_setup_logging, mock_mlflow, mock_setup_mlflow
    ):
        """Test train_example returns metrics dictionary."""
        mock_setup_mlflow.return_value = "test-experiment"

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        with patch.object(mock_mlflow, "start_run") as mock_start_run:
            mock_start_run.__enter__ = MagicMock(return_value=mock_run)
            mock_start_run.__exit__ = MagicMock(return_value=False)

            metrics = train_example()

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    @patch("src.train_example.setup_mlflow")
    @patch("src.train_example.mlflow")
    @patch("src.train_example.setup_logging")
    def test_train_example_custom_experiment(
        self, mock_setup_logging, mock_mlflow, mock_setup_mlflow
    ):
        """Test train_example with custom experiment name."""
        mock_setup_mlflow.return_value = "custom-experiment"

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        with patch.object(mock_mlflow, "start_run") as mock_start_run:
            mock_start_run.__enter__ = MagicMock(return_value=mock_run)
            mock_start_run.__exit__ = MagicMock(return_value=False)

            train_example(experiment_name="custom-experiment")

        mock_setup_mlflow.assert_called_once()

    @patch("src.train_example.setup_mlflow")
    @patch("src.train_example.mlflow")
    @patch("src.train_example.setup_logging")
    def test_train_example_staging_environment(
        self, mock_setup_logging, mock_mlflow, mock_setup_mlflow
    ):
        """Test train_example with staging environment."""
        mock_setup_mlflow.return_value = "test-experiment"

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.active_run.return_value = mock_run

        with patch.object(mock_mlflow, "start_run") as mock_start_run:
            mock_start_run.__enter__ = MagicMock(return_value=mock_run)
            mock_start_run.__exit__ = MagicMock(return_value=False)

            train_example(environment=Environment.STAGING)

        call_kwargs = mock_setup_mlflow.call_args[1]
        assert call_kwargs["environment"] == Environment.STAGING

    @patch("src.train_example.setup_mlflow")
    @patch("src.train_example.setup_logging")
    def test_train_example_config_error(
        self, mock_setup_logging, mock_setup_mlflow
    ):
        """Test train_example raises TrainingError on config error."""
        mock_setup_mlflow.side_effect = MLflowConfigError("Config error")

        with pytest.raises(MLflowConfigError):
            train_example()

    @patch("src.train_example.setup_mlflow")
    @patch("src.train_example.setup_logging")
    def test_train_example_logs_errors(
        self, mock_setup_logging, mock_setup_mlflow, caplog
    ):
        """Test train_example logs errors properly."""
        mock_setup_mlflow.side_effect = MLflowConfigError("Config error")

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(MLflowConfigError),
        ):
            train_example()

        assert "Config error" in caplog.text


class TestModuleConstants:
    """Tests for module-level constants and imports."""

    def test_environment_enum_imported(self):
        """Test Environment is properly imported."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_max_port_constant_accessible(self):
        """Test MAX_PORT constant is accessible."""
        from src.config.mlflow_config import MAX_PORT

        assert MAX_PORT == 65535


class TestErrorHandling:
    """Tests for error handling paths."""

    @patch("src.train_example.load_iris")
    def test_load_data_exception(self, mock_load_iris):
        """Test load_data handles exceptions."""
        mock_load_iris.side_effect = RuntimeError("Failed to load")

        with pytest.raises(TrainingError, match="Falha ao carregar dados"):
            load_data()

    def test_train_model_exception(self):
        """Test train_model handles exceptions."""
        X_train = "invalid"
        y_train = "invalid"

        with pytest.raises(TrainingError, match="Falha no treinamento"):
            train_model(X_train, y_train)  # type: ignore

    def test_evaluate_model_exception(self):
        """Test evaluate_model handles exceptions."""
        # Create a mock model that raises error
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Prediction failed")

        X_test = MagicMock()
        y_test = MagicMock()

        with pytest.raises(TrainingError, match="Falha na avaliação"):
            evaluate_model(mock_model, X_test, y_test)

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_no_active_run(self, mock_mlflow):
        """Test log_to_mlflow handles no active run."""
        mock_mlflow.active_run.return_value = None

        model = MagicMock()
        params = {"n_estimators": 100}
        metrics = {"accuracy": 0.9}

        with pytest.raises(MLflowConfigError, match="Nenhum run ativo"):
            log_to_mlflow(model, params, metrics)

    @patch("src.train_example.mlflow")
    def test_log_to_mlflow_exception(self, mock_mlflow):
        """Test log_to_mlflow handles MlflowException."""
        from mlflow.exceptions import MlflowException

        mock_mlflow.log_params.side_effect = MlflowException("MLflow error")

        model = MagicMock()
        params = {"n_estimators": 100}
        metrics = {"accuracy": 0.9}

        with pytest.raises(MLflowConfigError, match="Falha ao logar"):
            log_to_mlflow(model, params, metrics)

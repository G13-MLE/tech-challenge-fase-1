"""Tests for train_example module."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config.mlflow_config import MLflowConfigError
from src.train_example import (
    TrainingError,
    evaluate_model,
    load_data,
    log_to_mlflow,
    train_example,
    train_model,
)

# Constants for Iris dataset
IRIS_SAMPLES = 150
IRIS_FEATURES = 4
IRIS_TRAIN_SAMPLES_70 = 105
IRIS_TEST_SAMPLES_30 = 45
IRIS_TRAIN_SAMPLES_50 = 75
IRIS_TEST_SAMPLES_50 = 75

# Constants for model parameters
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 10
CUSTOM_N_ESTIMATORS_50 = 50
CUSTOM_MAX_DEPTH_5 = 5
CUSTOM_N_ESTIMATORS_200 = 200

# Constants for accuracy threshold
ACCURACY_THRESHOLD = 0.9


class TestTrainingError:
    """Tests for TrainingError exception."""

    @staticmethod
    def test_training_error_message():
        """Test TrainingError can be raised with message."""
        msg = "Test training error"
        with pytest.raises(TrainingError, match=msg):
            raise TrainingError(msg)

    @staticmethod
    def test_training_error_inheritance():
        """Test TrainingError inherits from Exception."""
        assert issubclass(TrainingError, Exception)


class TestLoadData:
    """Tests for load_data function."""

    @staticmethod
    def test_load_data_returns_dict():
        """Test load_data returns dictionary."""
        data = load_data()

        assert isinstance(data, dict)
        assert "X_train" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_test" in data

    @staticmethod
    def test_load_data_shapes():
        """Test load_data returns correct shapes."""
        data = load_data(test_size=0.3, random_state=42)

        # Iris has 150 samples
        total_samples = data["X_train"].shape[0] + data["X_test"].shape[0]
        assert total_samples == IRIS_SAMPLES

        # With test_size=0.3, train should have ~105 samples
        assert data["X_train"].shape[0] == IRIS_TRAIN_SAMPLES_70
        assert data["X_test"].shape[0] == IRIS_TEST_SAMPLES_30

    @staticmethod
    def test_load_data_custom_params():
        """Test load_data with custom parameters."""
        data = load_data(test_size=0.5, random_state=123)

        assert data["X_train"].shape[0] == IRIS_TRAIN_SAMPLES_50
        assert data["X_test"].shape[0] == IRIS_TEST_SAMPLES_50

    @staticmethod
    def test_load_data_feature_shapes():
        """Test load_data feature dimensions."""
        data = load_data()

        # Iris has 4 features
        assert data["X_train"].shape[1] == IRIS_FEATURES
        assert data["X_test"].shape[1] == IRIS_FEATURES

    @staticmethod
    def test_load_data_reproducibility():
        """Test load_data produces same results with same random_state."""
        data1 = load_data(random_state=42)
        data2 = load_data(random_state=42)

        np.testing.assert_array_equal(data1["X_train"], data2["X_train"])
        np.testing.assert_array_equal(data1["y_train"], data2["y_train"])


@pytest.fixture
def sample_data():
    """Create sample training data."""
    data = load_data()
    return data["X_train"], data["y_train"]


class TestTrainModel:
    """Tests for train_model function."""

    @staticmethod
    def test_train_model_returns_model(sample_data):
        """Test train_model returns RandomForestClassifier."""
        X_train, y_train = sample_data
        model = train_model(X_train, y_train)

        assert isinstance(model, RandomForestClassifier)

    @staticmethod
    def test_train_model_default_params(sample_data):
        """Test train_model with default parameters."""
        X_train, y_train = sample_data
        model = train_model(X_train, y_train)

        assert model.n_estimators == DEFAULT_N_ESTIMATORS
        assert model.max_depth == DEFAULT_MAX_DEPTH

    @staticmethod
    def test_train_model_custom_params(sample_data):
        """Test train_model with custom parameters."""
        X_train, y_train = sample_data
        model = train_model(
            X_train,
            y_train,
            params={
                "n_estimators": CUSTOM_N_ESTIMATORS_50,
                "max_depth": CUSTOM_MAX_DEPTH_5,
                "random_state": 123,
            },
        )

        assert model.n_estimators == CUSTOM_N_ESTIMATORS_50
        assert model.max_depth == CUSTOM_MAX_DEPTH_5

    @staticmethod
    def test_train_model_fitted(sample_data):
        """Test train_model returns fitted model."""
        X_train, y_train = sample_data
        model = train_model(X_train, y_train)

        # Check model has been fitted by predicting
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)

    @staticmethod
    def test_train_model_merges_params(sample_data):
        """Test train_model merges custom params with defaults."""
        X_train, y_train = sample_data
        model = train_model(
            X_train,
            y_train,
            params={"n_estimators": CUSTOM_N_ESTIMATORS_200},
        )

        assert model.n_estimators == CUSTOM_N_ESTIMATORS_200
        # Default max_depth should still be 10
        assert model.max_depth == DEFAULT_MAX_DEPTH


@pytest.fixture
def trained_model_and_data():
    """Create trained model and test data."""
    data = load_data()
    model = train_model(data["X_train"], data["y_train"])
    return model, data["X_test"], data["y_test"]


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @staticmethod
    def test_evaluate_model_returns_dict(trained_model_and_data):
        """Test evaluate_model returns dictionary."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert isinstance(metrics, dict)

    @staticmethod
    def test_evaluate_model_keys(trained_model_and_data):
        """Test evaluate_model returns correct keys."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    @staticmethod
    def test_evaluate_model_values_range(trained_model_and_data):
        """Test evaluate_model metrics are in valid range."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        for metric_name, metric_value in metrics.items():
            assert 0.0 <= metric_value <= 1.0, (
                f"{metric_name} out of range: {metric_value}"
            )

    @staticmethod
    def test_evaluate_model_high_accuracy(trained_model_and_data):
        """Test evaluate_model achieves high accuracy on Iris."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        # Random forest on Iris should achieve > 90% accuracy
        assert metrics["accuracy"] > ACCURACY_THRESHOLD


@pytest.fixture
def trained_model_and_metrics():
    """Create trained model and metrics."""
    data = load_data()
    model = train_model(data["X_train"], data["y_train"])
    metrics = evaluate_model(model, data["X_test"], data["y_test"])
    params = {
        "n_estimators": DEFAULT_N_ESTIMATORS,
        "max_depth": DEFAULT_MAX_DEPTH,
        "random_state": 42,
    }
    return model, params, metrics


@patch("src.train_example.mlflow")
def test_log_to_mlflow_logs_params(mock_mlflow, trained_model_and_metrics):
    """Test log_to_mlflow logs parameters."""
    model, params, metrics = trained_model_and_metrics

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.active_run.return_value = mock_run

    log_to_mlflow(model, params, metrics)

    mock_mlflow.log_params.assert_called_once_with(params)


@patch("src.train_example.mlflow")
def test_log_to_mlflow_logs_metrics(mock_mlflow, trained_model_and_metrics):
    """Test log_to_mlflow logs metrics."""
    model, params, metrics = trained_model_and_metrics

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.active_run.return_value = mock_run

    log_to_mlflow(model, params, metrics)

    assert mock_mlflow.log_metric.call_count == len(metrics)


@patch("src.train_example.mlflow")
def test_log_to_mlflow_logs_model(mock_mlflow, trained_model_and_metrics):
    """Test log_to_mlflow logs model."""
    model, params, metrics = trained_model_and_metrics

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.active_run.return_value = mock_run

    log_to_mlflow(model, params, metrics)

    mock_mlflow.sklearn.log_model.assert_called_once()


@patch("src.train_example.mlflow")
def test_log_to_mlflow_with_tags(mock_mlflow, trained_model_and_metrics):
    """Test log_to_mlflow with tags."""
    model, params, metrics = trained_model_and_metrics
    tags = {"version": "v1.0.0", "author": "test"}

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.active_run.return_value = mock_run

    log_to_mlflow(model, params, metrics, tags=tags)

    assert mock_mlflow.set_tag.call_count == len(tags)


@patch("src.train_example.mlflow")
def test_log_to_mlflow_returns_run_id(mock_mlflow, trained_model_and_metrics):
    """Test log_to_mlflow returns run ID."""
    model, params, metrics = trained_model_and_metrics

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.active_run.return_value = mock_run

    run_id = log_to_mlflow(model, params, metrics)

    assert run_id == "test-run-id"


@patch("src.train_example.setup_mlflow")
@patch("src.train_example.mlflow")
@patch("src.train_example.setup_logging")
def test_train_example_returns_metrics(
    mock_setup_logging, mock_mlflow, mock_setup_mlflow
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
    mock_setup_logging, mock_mlflow, mock_setup_mlflow
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

    mock_setup_mlflow.assert_called_once_with(
        experiment_name="custom-experiment"
    )


@patch("src.train_example.setup_mlflow")
@patch("src.train_example.setup_logging")
def test_train_example_config_error(mock_setup_logging, mock_setup_mlflow):
    """Test train_example raises TrainingError on config error."""

    mock_setup_mlflow.side_effect = MLflowConfigError("Config error")

    with pytest.raises(TrainingError, match="Erro inesperado"):
        train_example()


@patch("src.train_example.setup_mlflow")
@patch("src.train_example.setup_logging")
def test_train_example_logs_errors(
    mock_setup_logging, mock_setup_mlflow, caplog
):
    """Test train_example logs errors properly."""

    mock_setup_mlflow.side_effect = MLflowConfigError("Config error")

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(TrainingError),
    ):
        train_example()

    assert "Config error" in caplog.text


@patch("src.train_example.load_iris")
def test_load_data_exception(mock_load_iris):
    """Test load_data raises exception."""
    mock_load_iris.side_effect = RuntimeError("Failed to load")

    # load_data no longer wraps exceptions in TrainingError
    # it lets exceptions bubble up
    with pytest.raises(RuntimeError, match="Failed to load"):
        load_data()


@patch("src.train_example.mlflow")
def test_log_to_mlflow_no_active_run(mock_mlflow):
    """Test log_to_mlflow handles no active run."""
    mock_mlflow.active_run.return_value = None

    model = MagicMock()
    params = {"n_estimators": DEFAULT_N_ESTIMATORS}
    metrics = {"accuracy": ACCURACY_THRESHOLD}

    with pytest.raises(TrainingError, match="Nenhum run ativo"):
        log_to_mlflow(model, params, metrics)

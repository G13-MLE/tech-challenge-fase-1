"""Testes para o pipeline Dummy Baseline.

Testa a orquestracao do pipeline e a integracao com
o modulo de treinamento de dummy classifiers.
"""

from __future__ import annotations

import os
from typing import Self

import pandas as pd
import pytest

from src.data.splitting import split_train_test_stratified
from src.data.validation import validate_required_columns
from src.pipelines.run_dummy_baseline import main
from src.training import DummyTrainingConfig
from src.training.dummy_trainer import run_all_strategies
from src.training.metrics import compute_binary_classification_metrics
from src.training.mlflow_tracking import MLflowConfig, setup_mlflow

POSITIVE_LABEL = "Yes"

EXPECTED_SPLIT_SIZE = 3
EXPECTED_STRATEGY_COUNT = 3


def make_dummy_df() -> pd.DataFrame:
    """Cria dataset minimo para validar pipeline dummy."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 1],
            "Churn": ["No", "Yes", "No", "Yes", "No", "Yes"],
        }
    )


def test_split_data_returns_expected_sizes() -> None:
    """Split deve retornar conjuntos com tamanhos consistentes."""
    df = make_dummy_df()
    config = DummyTrainingConfig(test_size=0.5, random_seed=42)

    X_train, X_test, y_train, y_test = split_train_test_stratified(
        df,
        config.target_column,
        config.test_size,
        config.random_seed,
    )

    assert len(X_train) == EXPECTED_SPLIT_SIZE
    assert len(X_test) == EXPECTED_SPLIT_SIZE
    assert len(y_train) == EXPECTED_SPLIT_SIZE
    assert len(y_test) == EXPECTED_SPLIT_SIZE


def test_compute_metrics_returns_all_expected_keys() -> None:
    """Calculo de metricas deve retornar todas as chaves esperadas."""
    y_true = pd.Series([POSITIVE_LABEL, "No", POSITIVE_LABEL, "No"])
    y_pred = pd.Series([POSITIVE_LABEL, "No", "No", "No"])
    y_proba = pd.Series([0.8, 0.1, 0.3, 0.2])

    metrics = compute_binary_classification_metrics(
        y_true,
        y_pred,
        y_proba,
        POSITIVE_LABEL,
    )

    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
    }


def test_run_all_strategies_returns_three_rows(monkeypatch: object) -> None:
    """Pipeline deve produzir 3 resultados, um por estrategia dummy."""
    df = make_dummy_df()
    config = DummyTrainingConfig(test_size=0.5, random_seed=42)
    X_train, X_test, y_train, y_test = split_train_test_stratified(
        df,
        config.target_column,
        config.test_size,
        config.random_seed,
    )

    # Mock do MLflow
    class _DummyRun:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.start_run",
        lambda run_name=None: _DummyRun(),
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.set_tag",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_input",
        lambda *args, **kwargs: None,
    )

    results_df = run_all_strategies(
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        dataset_version="abc123",
    )

    assert len(results_df) == EXPECTED_STRATEGY_COUNT
    assert set(results_df["strategy"].tolist()) == {
        "most_frequent",
        "stratified",
        "uniform",
    }


def test_validate_required_columns_raises_for_missing_target() -> None:
    """Validacao deve falhar quando target nao existe no dataframe."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    with pytest.raises(ValueError, match="Coluna alvo ausente"):
        validate_required_columns(df, "Churn")


def test_setup_mlflow_sets_local_env(monkeypatch: object) -> None:
    """Configuracao local deve preencher variaveis de ambiente MinIO."""
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    monkeypatch.setattr(
        "src.training.mlflow_tracking.mlflow.set_tracking_uri",
        lambda _: None,
    )
    monkeypatch.setattr(
        "src.training.mlflow_tracking.mlflow.set_experiment",
        lambda _: None,
    )

    setup_mlflow(MLflowConfig(tracking_uri="http://localhost:5000"))

    assert os.environ["MLFLOW_S3_ENDPOINT_URL"] == "http://localhost:9000"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "minioadmin"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "minioadmin_secret_key_2024"


def test_main_returns_zero_with_monkeypatched_flow(
    monkeypatch: object,
) -> None:
    """main deve concluir com sucesso quando dependencias sao mockadas."""
    df = make_dummy_df()
    monkeypatch.setattr(
        "src.pipelines.run_dummy_baseline.load_telco_data",
        lambda: df,
    )
    monkeypatch.setattr(
        "src.pipelines.run_dummy_baseline.setup_mlflow",
        lambda _: None,
    )
    monkeypatch.setattr(
        "src.pipelines.run_dummy_baseline.setup_logging",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.pipelines.run_dummy_baseline.run_all_strategies",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "strategy": ["uniform"],
                "accuracy": [0.5],
                "precision": [0.5],
                "recall": [0.5],
                "f1_score": [0.5],
                "roc_auc": [0.5],
                "pr_auc": [0.5],
            }
        ),
    )

    assert main() == 0

"""Utilitários genéricos para tracking com MLflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mlflow
import numpy as np
import pandas as pd
from mlflow.data import from_pandas  # type: ignore[attr-defined]

DEFAULT_DATASET_SOURCE_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@dataclass(frozen=True)
class MLflowConfig:
    """Configuração de tracking do MLflow.

    Usa default_factory para lazy evaluation, garantindo que
    as variáveis de ambiente sejam lidas após carregar o .env.
    """

    tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
    )
    experiment_name: str = "tech-challenge-default"


@dataclass(frozen=True)
class TrainTestData:
    """Estrutura com split de treino e teste para lineage.

    Suporta tanto pandas Series quanto numpy arrays para targets,
    facilitando integração com diferentes pipelines (sklearn usa
    Series, PyTorch usa arrays).
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | np.ndarray
    y_test: pd.Series | np.ndarray


def setup_mlflow(config: MLflowConfig) -> None:
    """Configura tracking URI e experimento no MLflow."""
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    tracking_uri = config.tracking_uri
    if "localhost" in tracking_uri:
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
        )
        os.environ.setdefault(
            "AWS_ACCESS_KEY_ID", os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        )
        os.environ.setdefault(
            "AWS_SECRET_ACCESS_KEY",
            os.getenv("MINIO_SECRET_KEY", "minioadmin_secret_key_2024"),
        )


def build_mlflow_inputs(
    train_test_data: TrainTestData,
    target_column: str,
    dataset_version: str,
    dataset_source_path: str = DEFAULT_DATASET_SOURCE_PATH,
) -> tuple[object, object]:
    """Cria datasets de input para lineage sem digest fixo manual.

    Converte automaticamente y_train/y_test para Series se
    forem arrays numpy, garantindo compatibilidade com MLflow.

    Returns:
        Tupla de (train_input, test_input) como objetos Dataset MLflow.
        Retorna object pois o tipo específico não está disponível em
        type stubs.
    """
    dataset_version_short = dataset_version[:8]

    # Converte arrays para Series se necessário
    y_train_series: pd.Series
    y_test_series: pd.Series

    if isinstance(train_test_data.y_train, np.ndarray):
        y_train_series = pd.Series(train_test_data.y_train)
    else:
        y_train_series = train_test_data.y_train

    if isinstance(train_test_data.y_test, np.ndarray):
        y_test_series = pd.Series(train_test_data.y_test)
    else:
        y_test_series = train_test_data.y_test

    train_dataset = train_test_data.X_train.copy()
    train_dataset[target_column] = y_train_series
    test_dataset = train_test_data.X_test.copy()
    test_dataset[target_column] = y_test_series

    train_input = from_pandas(
        train_dataset,
        source=dataset_source_path,
        name=f"train_split_v{dataset_version_short}",
    )
    test_input = from_pandas(
        test_dataset,
        source=dataset_source_path,
        name=f"test_split_v{dataset_version_short}",
    )
    return train_input, test_input

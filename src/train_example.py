"""
Script de exemplo para validar o tracking doMLflow.

Treina um modelo RandomForest no dataset Iris e loga
metricas, parametros e modelo noMLflow.

Usage:
    python src/train_example.py
    uv run python src/train_example.py

Environment Variables:
   MLFLOW_TRACKING_URI: URI do servidorMLflow
   MLFLOW_EXPERIMENT_NAME: Nome do experimento
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.mlflow_config import setup_logging, setup_mlflow

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Excecao para erros durante treinamento."""


def load_data(
    test_size: float = 0.2, random_state: int = 42
) -> dict[str, Any]:
    """
    Carrega e divide o dataset Iris.

    Args:
        test_size: Proporcao do dataset para teste
        random_state: Seed para reprodutibilidade

    Returns:
        Dict com X_train, X_test, y_train, y_test
    """
    logger.info("Carregando dataset Iris...")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Dados carregados: {X_train.shape[0]} treino, {X_test.shape[0]} teste"
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def train_model(
    X_train: np.ndarray[Any, np.dtype[Any]],
    y_train: np.ndarray[Any, np.dtype[Any]],
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """
    Treina um modelo RandomForest.

    Args:
        X_train: Features de treino
        y_train: Labels de treino
        params: Parametros do modelo (opcional)

    Returns:
        Modelo treinado
    """
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42,
    }

    model_params = {**default_params, **(params or {})}

    logger.info(f"Treinando modelo com parametros: {model_params}")

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    logger.info("Modelo treinado com sucesso")

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray[Any, np.dtype[Any]],
    y_test: np.ndarray[Any, np.dtype[Any]],
) -> dict[str, float]:
    """
    Avalia o modelo e calcula metricas.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Labels de teste

    Returns:
        Dict com metricas calculadas
    """
    logger.info("Avaliando modelo...")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    logger.info(f"Metricas calculadas: {metrics}")
    return metrics


def log_to_mlflow(
    model: RandomForestClassifier,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> str:
    """
    Loga modelo, parametros e metricas no MLflow.

    Args:
        model: Modelo treinado
        params: Parametros do modelo
        metrics: Metricas calculadas
        tags: Tags adicionais (opcional)

    Returns:
        ID do run
    """
    logger.info("Logando artefatos no MLflow...")

    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForestClassifier")

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=None,
    )

    if tags:
        for tag_name, tag_value in tags.items():
            mlflow.set_tag(tag_name, tag_value)

    active_run = mlflow.active_run()
    if active_run is None:
        raise TrainingError("Nenhum run ativo encontrado")

    run_id = active_run.info.run_id
    logger.info(f"Run ID: {run_id}")

    return run_id


def train_example(
    experiment_name: str = "validacao-mlflow",
) -> dict[str, float]:
    """
    Pipeline completo de treino com MLflow tracking.

    Args:
        experiment_name: Nome do experimento

    Returns:
        Dict com metricas do modelo

    Example:
        >>> metrics = train_example()
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    """
    setup_logging(logging.INFO)

    logger.info("=" * 60)
    logger.info("Iniciando pipeline de treinamento")
    logger.info("=" * 60)

    try:
        setup_mlflow(experiment_name=experiment_name)
        logger.info(f"Experimento: {experiment_name}")

        data = load_data(test_size=0.2, random_state=42)

        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
        }

        with mlflow.start_run(run_name="baseline-random-forest"):
            logger.info("Run iniciada")

            model = train_model(data["X_train"], data["y_train"], params)

            metrics = evaluate_model(model, data["X_test"], data["y_test"])

            tags = {
                "versao": "v0.2.0",
                "autor": "tech-challenge",
                "framework": "sklearn",
            }

            log_to_mlflow(model, params, metrics, tags)

            logger.info("=" * 60)
            logger.info("Pipeline completado com sucesso!")
            logger.info("=" * 60)

            return metrics

    except Exception as e:
        logger.error(f"Erro no pipeline: {e}")
        raise TrainingError(f"Erro inesperado no pipeline: {e}") from e


def main() -> None:
    """Entry point para execucao via terminal."""
    parser = argparse.ArgumentParser(
        description="Treina modelo de exemplo comMLflow tracking"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        default="validacao-mlflow",
        help="Nome do experimento (default: validacao-mlflow)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Habilita logs detalhados"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    try:
        metrics = train_example(experiment_name=args.experiment)

        print("\n" + "=" * 60)
        print("METRICAS FINAIS")
        print("=" * 60)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("=" * 60)

        sys.exit(0)

    except (TrainingError, OSError) as e:
        logger.error(f"Execucao falhou: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

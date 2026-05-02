"""Treinador para modelo DummyClassifier baseline.

Este módulo fornece funções para treinar e avaliar múltiplas
estratégias do DummyClassifier com tracking no MLflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.constants import POSITIVE_LABEL, RANDOM_SEED
from src.training.metrics import compute_binary_classification_metrics

logger = logging.getLogger(__name__)

STRATEGIES = ("most_frequent", "stratified", "uniform")


@dataclass(frozen=True)
class DummyTrainingConfig:
    """Configuração para treino do DummyClassifier."""

    test_size: float = 0.2
    random_seed: int = RANDOM_SEED
    target_column: str = "Churn"


def train_dummy_strategy(  # noqa: PLR0913, PLR0917
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    strategy: str,
    config: DummyTrainingConfig,
) -> dict[str, Any]:
    """Treina e avalia uma única estratégia do DummyClassifier.

    Args:
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
        strategy: Estratégia do DummyClassifier
            (most_frequent, stratified, uniform)
        config: Configuração do treinamento

    Returns:
        Dicionário com resultado da estratégia incluindo métricas
    """
    model = DummyClassifier(
        strategy=strategy,
        random_state=config.random_seed,
    )
    model.fit(X_train, y_train)

    y_pred = pd.Series(
        model.predict(X_test),
        index=y_test.index,
    ).astype(str)

    proba_classes = list(model.classes_)
    positive_idx = proba_classes.index(POSITIVE_LABEL)
    y_proba_positive = pd.Series(
        model.predict_proba(X_test)[:, positive_idx],
        index=y_test.index,
    )

    metrics = compute_binary_classification_metrics(
        y_test,
        y_pred,
        y_proba_positive,
        POSITIVE_LABEL,
    )

    return {
        "strategy": strategy,
        "model": model,
        "metrics": metrics,
    }


def run_all_strategies(  # noqa: PLR0913, PLR0917
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: DummyTrainingConfig,
    dataset_version: str = "unknown",
    train_input: Any | None = None,
    test_input: Any | None = None,
) -> pd.DataFrame:
    """Executa treino/eval/log para cada estrategia do DummyClassifier.

    Args:
        X_train: Features de treino.
        X_test: Features de teste.
        y_train: Target de treino.
        y_test: Target de teste.
        config: Configuracao do treinamento.
        dataset_version: Versao do dataset para tracking.
        train_input: Input MLflow para dados de treino (opcional).
        test_input: Input MLflow para dados de teste (opcional).

    Returns:
        DataFrame comparativo com metricas por estratégia,
        ordenado por F1 score.
    """
    results: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        result = train_dummy_strategy(
            X_train, X_test, y_train, y_test, strategy, config
        )
        metrics = result["metrics"]

        run_name = f"dummy_{strategy}"
        with mlflow.start_run(run_name=run_name):
            if train_input is not None:
                mlflow.log_input(train_input, context="training")  # type: ignore[arg-type]
            if test_input is not None:
                mlflow.log_input(test_input, context="testing")  # type: ignore[arg-type]

            mlflow.log_param("model_type", "DummyClassifier")
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("random_seed", config.random_seed)
            mlflow.log_param("test_size", config.test_size)
            mlflow.log_param("target_column", config.target_column)
            mlflow.log_param("dataset_version", dataset_version)

            mlflow.set_tag("issue", "20")
            mlflow.set_tag("baseline_family", "dummy")
            mlflow.set_tag("model_baseline", "dummy_classifier")
            mlflow.set_tag("random_seed", str(config.random_seed))

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        results.append({"strategy": strategy, **metrics})

        logger.info(
            "Estrategia %s: accuracy=%.4f f1=%.4f",
            strategy,
            metrics["accuracy"],
            metrics["f1_score"],
        )

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False,
    )

    return results_df

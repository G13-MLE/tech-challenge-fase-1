"""
Configuracao simplificada do MLflow para o projeto.

Pratica recomendada: usar um unico servidorMLflow centralizado.
A transicao entre staging e production e feita via Tags e Aliases
no Model Registry, nao via backends separados.

Usage:
    from src.config.mlflow_config import setup_mlflow

    # Configurar MLflow (usaMLFLOW_TRACKING_URI do ambiente)
    setup_mlflow(experiment_name="meu-experimento")

    # Ou com URI explicita
    setup_mlflow(
        experiment_name="meu-experimento",
        tracking_uri="http://localhost:5000"
    )
"""

import logging
import os

import mlflow


def setup_mlflow(
    experiment_name: str = "tech-challenge-fase-1",
    tracking_uri: str | None = None,
) -> str:
    """
    Configura o MLflow para o projeto.

    Args:
        experiment_name: Nome do experimento
        tracking_uri: URI do servidorMLflow (opcional)
            Se nao informado, usaMLFLOW_TRACKING_URI do ambiente
            ou file:./mlruns como fallback

    Returns:
        Nome do experimento configurado

    Example:
        >>> setup_mlflow(experiment_name="meu-experimento")
        'meu-experimento'
    """
    uri: str
    if tracking_uri:
        uri = tracking_uri
    else:
        uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    logger = logging.getLogger(__name__)
    logger.info(
        f"MLflow configurado - tracking_uri: {uri}, "
        f"experiment: {experiment_name}"
    )

    return experiment_name


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura logging para o projeto.

    Args:
        level: Nivel de logging (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


class MLflowConfigError(Exception):
    """Excecao para erros de configuracao doMLflow."""

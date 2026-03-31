"""
Configuracao simplificada do MLflow para o projeto Tech Challenge Fase 1.

Este modulo fornece configuracao basica do MLflow usando um unico servidor
centralizado. A transicao entre staging e production e feita via Tags e
Aliases no Model Registry.

Example:
    >>> from src.config import setup_mlflow
    >>>
    >>> # Configuracao basica
    >>> setup_mlflow()
    >>>
    >>> # Com experimento especifico
    >>> setup_mlflow(experiment_name="meu-experimento")
"""

from src.config.mlflow_config import (
    MLflowConfigError,
    setup_logging,
    setup_mlflow,
)

__all__ = [
    "MLflowConfigError",
    "setup_logging",
    "setup_mlflow",
]

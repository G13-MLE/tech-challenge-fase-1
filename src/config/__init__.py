"""
Configuração do MLflow para o projeto Tech Challenge Fase 1.

Este módulo fornece configuração centralizada do MLflow com suporte
a múltiplos ambientes (development, staging, production).

Example:
    >>> from src.config import setup_mlflow, Environment
    >>>
    >>> # Configuração para desenvolvimento
    >>> setup_mlflow()
    >>>
    >>> # Configuração para produção
    >>> setup_mlflow(environment=Environment.PRODUCTION)
"""

from src.config.mlflow_config import (
    Environment,
    MLflowConfig,
    MLflowConfigError,
    setup_mlflow,
    setup_logging,
    get_mlflow_port,
)

__all__ = [
    "Environment",
    "MLflowConfig",
    "MLflowConfigError",
    "setup_mlflow",
    "setup_logging",
    "get_mlflow_port",
]

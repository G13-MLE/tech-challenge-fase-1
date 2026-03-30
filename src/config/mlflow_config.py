"""
Configuração centralizada do MLflow para o projeto.
Baseado nas práticas ensinadas nas aulas de MLflow.

Supports multiple environments: development, staging, production.
"""

import os
import logging
from enum import Enum
from typing import Optional
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException


class Environment(Enum):
    """Ambientes de execução do projeto."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class MLflowConfigError(Exception):
    """Exceção customizada para erros de configuração do MLflow."""

    pass


class MLflowConfig:
    """
    Configuração do MLflow com suporte a múltiplos ambientes.

    Attributes:
        tracking_uri: URI do servidor de tracking
        experiment_name: Nome padrão do experimento
        port: Porta para MLflow UI
        artifact_root: Diretório raiz para artefatos

    Example:
        >>> config = MLflowConfig.for_env(Environment.DEVELOPMENT)
        >>> experiment = config.setup()
        >>> print(config.tracking_uri)
        file:./mlruns
    """

    _DEFAULTS = {
        Environment.DEVELOPMENT: {
            "tracking_uri": "file:./mlruns",
            "artifact_root": "./mlruns",
            "port": 5000,
            "host": "127.0.0.1",
        },
        Environment.STAGING: {
            "tracking_uri": os.getenv(
                "MLFLOW_TRACKING_URI", "sqlite:///mlflow-staging.db"
            ),
            "artifact_root": "./mlruns-staging",
            "port": 5001,
            "host": "0.0.0.0",
        },
        Environment.PRODUCTION: {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "artifact_root": os.getenv(
                "MLFLOW_ARTIFACT_ROOT", "s3://mlflow-artifacts/"
            ),
            "port": int(os.getenv("MLFLOW_PORT", "5000")),
            "host": os.getenv("MLFLOW_HOST", "0.0.0.0"),
        },
    }

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        port: Optional[int] = None,
        host: Optional[str] = None,
        artifact_root: Optional[str] = None,
        environment: Environment = Environment.DEVELOPMENT,
    ):
        self.logger = logging.getLogger(__name__)

        defaults = self._DEFAULTS.get(
            environment, self._DEFAULTS[Environment.DEVELOPMENT]
        )

        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", defaults["tracking_uri"]
        )
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "tech-challenge-fase-1"
        )
        self.port = port or int(
            os.getenv("MLFLOW_PORT", str(defaults.get("port", 5000)))
        )
        self.host = host or os.getenv("MLFLOW_HOST", defaults.get("host", "127.0.0.1"))
        self.artifact_root = artifact_root or os.getenv(
            "MLFLOW_ARTIFACT_ROOT", defaults.get("artifact_root", "./mlruns")
        )
        self.environment = environment

        self._validate_config()

    @classmethod
    def for_env(cls, environment: Environment = Environment.DEVELOPMENT, **kwargs):
        """
        Factory method para criar configuração para um ambiente específico.

        Args:
            environment: Ambiente de execução (DEVELOPMENT, STAGING, PRODUCTION)
            **kwargs: Argumentos adicionais para sobrescrever defaults

        Returns:
            MLflowConfig configurado para o ambiente

        Example:
            >>> config = MLflowConfig.for_env(Environment.PRODUCTION)
        """
        return cls(environment=environment, **kwargs)

    @classmethod
    def from_env_vars(cls):
        """
        Cria configuração a partir de variáveis de ambiente.

        Returns:
            MLflowConfig com valores das variáveis de ambiente
        """
        return cls()

    def _validate_config(self):
        """Valida a configuração do MLflow."""
        if not self.tracking_uri:
            raise MLflowConfigError("tracking_uri é obrigatório")

        if not self.experiment_name:
            raise MLflowConfigError("experiment_name é obrigatório")

        if self.port < 1 or self.port > 65535:
            raise MLflowConfigError(
                f"Porta inválida: {self.port}. Deve ser entre 1 e65535"
            )

        if (
            self.environment == Environment.PRODUCTION
            and not self.tracking_uri.startswith(("http", "https", "s3", "gs", "azure"))
        ):
            self.logger.warning(
                "PRODUÇÃO: tracking_uri deve usar servidor remoto (http/https/s3/gs/azure). "
                f"Atual: {self.tracking_uri}"
            )

    def setup(self) -> str:
        """
        Configura o MLflow para o projeto.

        Returns:
            Nome do experimento configurado

        Raises:
            MLflowConfigError: Se a configuração for inválida
            MlflowException: Se houver erro na conexão comMLflow

        Example:
            >>> config = MLflowConfig.for_env(Environment.DEVELOPMENT)
            >>> experiment = config.setup()
        """
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.logger.info(f"Tracking URI configurado: {self.tracking_uri}")

            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"Experimento configurado: {self.experiment_name}")

            self.logger.info(
                f"MLflow configurado com sucesso - "
                f"Ambiente: {self.environment.value}, "
                f"Porta: {self.port}"
            )

            return self.experiment_name

        except MlflowException as e:
            self.logger.error(f"Erro ao configurar MLflow: {e}")
            raise MLflowConfigError(f"Falha ao configurar MLflow: {e}") from e
        except Exception as e:
            self.logger.error(f"Erro inesperado: {e}")
            raise MLflowConfigError(f"Erro inesperado na configuração: {e}") from e

    def get_artifact_path(self) -> Path:
        """
        Retorna o caminho para artefatos.

        Returns:
            Path objeto do diretório de artefatos
        """
        if self.tracking_uri.startswith("file:"):
            path = self.tracking_uri.replace("file:", "").lstrip("/")
            return Path(path).parent
        return Path(self.artifact_root)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura logging para o projeto.

    Args:
        level: Nível de logging (default: INFO)

    Example:
        >>> setup_logging(logging.DEBUG)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_mlflow(
    experiment_name: Optional[str] = None,
    environment: Environment = Environment.DEVELOPMENT,
) -> str:
    """
    Função de compatibilidade para configuração rápida do MLflow.

    Args:
        experiment_name: Nome do experimento (opcional)
        environment: Ambiente de execução (default: DEVELOPMENT)

    Returns:
        Nome do experimento configurado

    Raises:
        MLflowConfigError: Se a configuração for inválida

    Example:
        >>> from src.config.mlflow_config import setup_mlflow, Environment
        >>>
        >>> # Desenvolvimento (padrão)
        >>> setup_mlflow()
        >>>
        >>> # Produção
        >>> setup_mlflow(environment=Environment.PRODUCTION)
    """
    config = MLflowConfig.for_env(
        environment=environment, experiment_name=experiment_name
    )
    return config.setup()


def get_mlflow_port() -> int:
    """
    Retorna a porta configurada para o MLflow UI.

    Deprecated: Use MLflowConfig.port instead.

    Returns:
        Porta do MLflow UI
    """
    import warnings

    warnings.warn(
        "get_mlflow_port() está deprecado. Use MLflowConfig.port instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return int(os.getenv("MLFLOW_PORT", "5000"))


if __name__ == "__main__":
    setup_logging(logging.DEBUG)

    print("=" * 60)
    print("Testando configuração do MLflow")
    print("=" * 60)

    for env in Environment:
        print(f"\nAmbiente: {env.value}")
        config = MLflowConfig.for_env(env)
        print(f"  Tracking URI: {config.tracking_uri}")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Port: {config.port}")
        print(f"  Host: {config.host}")

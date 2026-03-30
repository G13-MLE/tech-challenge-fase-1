"""
Script de exemplo para validar o tracking do MLflow.
Demonstra as práticas ensinadas nas aulas.

Este script treina um modelo RandomForest no dataset Iris e loga
métricas, parâmetros e modelo no MLflow.

Usage:
    python src/train_example.py
    python -m src.train_example

Environment Variables:
    MLFLOW_TRACKING_URI: URI do servidor MLflow (default: file:./mlruns)
    MLFLOW_EXPERIMENT_NAME: Nome do experimento (default: tech-challenge-fase-1)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from src.config.mlflow_config import (
        setup_mlflow,
        setup_logging,
        MLflowConfig,
        Environment,
        MLflowConfigError,
    )
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config.mlflow_config import (
        setup_mlflow,
        setup_logging,
        MLflowConfig,
        Environment,
        MLflowConfigError,
    )


logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Exceção para erros durante treinamento."""

    pass


def load_data(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Carrega e divide o dataset Iris.

    Args:
        test_size: Proporção do dataset para teste (0.0 to 1.0)
        random_state: Seed para reprodutibilidade

    Returns:
        Dict com X_train, X_test, y_train, y_test

    Raises:
        TrainingError: Se falhar ao carregar dados
    """
    logger.info("Carregando dataset Iris...")

    try:
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
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise TrainingError(f"Falha ao carregar dados: {e}") from e


def train_model(
    X_train, y_train, params: Optional[Dict[str, Any]] = None
) -> RandomForestClassifier:
    """
    Treina um modelo RandomForest.

    Args:
        X_train: Features de treino
        y_train: Labels de treino
        params: Parâmetros do modelo (opcional)

    Returns:
        Modelo treinado

    Raises:
        TrainingError: Se falhar durante treinamento
    """
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42,
    }

    model_params = {**default_params, **params} if params else default_params

    logger.info(f"Treinando modelo com parâmetros: {model_params}")

    try:
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        logger.info("Modelo treinado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        raise TrainingError(f"Falha no treinamento: {e}") from e


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Avalia o modelo e calcula métricas.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Labels de teste

    Returns:
        Dict com métricas calculadas

    Raises:
        TrainingError: Se falhar na avaliação
    """
    logger.info("Avaliando modelo...")

    try:
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        logger.info(f"Métricas calculadas: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Erro na avaliação: {e}")
        raise TrainingError(f"Falha na avaliação: {e}") from e


def log_to_mlflow(
    model,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Loga modelo, parâmetros e métricas no MLflow.

    Args:
        model: Modelo treinado
        params: Parâmetros do modelo
        metrics: Métricas calculadas
        tags: Tags adicionais (opcional)

    Returns:
        ID do run

    Raises:
        MLflowConfigError: Se falhar ao logar no MLflow
    """
    logger.info("Logando artefatos no MLflow...")

    try:
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        logger.debug(f"Parâmetros logados: {params}")

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        logger.debug(f"Métricas logadas: {metrics}")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
        )
        logger.info("Modelo salvo no MLflow")

        if tags:
            for tag_name, tag_value in tags.items():
                mlflow.set_tag(tag_name, tag_value)
            logger.debug(f"Tags definidas: {tags}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        return run_id
    except MlflowException as e:
        logger.error(f"Erro MLflow: {e}")
        raise MLflowConfigError(f"Falha ao logar no MLflow: {e}") from e


def train_example(
    experiment_name: str = "validacao-mlflow",
    environment: Environment = Environment.DEVELOPMENT,
) -> Dict[str, float]:
    """
    Pipeline completo de treino com MLflow tracking.

    Executa: setup -> load_data -> train -> evaluate -> log

    Args:
        experiment_name: Nome do experimento
        environment: Ambiente de execução

    Returns:
        Dict com métricas do modelo

    Raises:
        TrainingError: Se falhar durante treinamento
        MLflowConfigError: Se falhar na configuração do MLflow

    Example:
        >>> from src.train_example import train_example
        >>> from src.config.mlflow_config import Environment
        >>>
        >>> metrics = train_example()
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    """
    setup_logging(logging.INFO)

    logger.info("=" * 60)
    logger.info("Iniciando pipeline de treinamento")
    logger.info("=" * 60)

    try:
        experiment = setup_mlflow(
            experiment_name=experiment_name, environment=environment
        )
        logger.info(f"Experimento: {experiment}")

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
                "environment": environment.value,
                "framework": "sklearn",
            }

            log_to_mlflow(model, params, metrics, tags)

            config = MLflowConfig.for_env(environment)
            logger.info(f"MLflow UI: http://localhost:{config.port}")

            logger.info("=" * 60)
            logger.info("Pipeline completado com sucesso!")
            logger.info("=" * 60)

            return metrics

    except (TrainingError, MLflowConfigError) as e:
        logger.error(f"Erro no pipeline: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise TrainingError(f"Erro inesperado no pipeline: {e}") from e


def main():
    """Entry point para execução via terminal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina modelo de exemplo com MLflow tracking"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        default="validacao-mlflow",
        help="Nome do experimento (default: validacao-mlflow)",
    )
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="development",
        help="Ambiente de execução (default: development)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Habilita logs detalhados"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    env_map = {
        "development": Environment.DEVELOPMENT,
        "staging": Environment.STAGING,
        "production": Environment.PRODUCTION,
    }

    try:
        metrics = train_example(
            experiment_name=args.experiment, environment=env_map[args.env]
        )

        print("\n" + "=" * 60)
        print("MÉTRICAS FINAIS")
        print("=" * 60)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Execução falhou: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

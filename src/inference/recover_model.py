"""Recuperacao de modelos treinados do MLflow.

Fornece funcoes para carregar modelos salvos no MLflow
para inference ou analise pos-treino.

Como usar (CLI):
    recover_model --model-type mlp --output models/recovered
    recover_model --model-type logistic --output models/recovered
    recover_model --model-type dummy --strategy most_frequent
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.api.logging import setup_logging
from src.constants import (
    DEFAULT_DUMMY_EXPERIMENT_NAME,
    DEFAULT_LOGISTIC_EXPERIMENT_NAME,
    DEFAULT_MLP_EXPERIMENT_NAME,
)
from src.pipelines.common import load_dotenv_silent

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("mlp", "logistic", "dummy")


def get_latest_run_id(
    experiment_name: str,
) -> str:
    """Retorna o run_id mais recente de um experimento.

    Args:
        experiment_name: Nome do experimento no MLflow.

    Returns:
        ID do run mais recente.

    Raises:
        RuntimeError: Se nenhum run for encontrado.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        msg = (
            f"Experimento '{experiment_name}' nao encontrado. "
            "Execute o treinamento primeiro."
        )
        raise RuntimeError(msg)

    runs: pd.DataFrame = mlflow.search_runs(  # type: ignore[assignment]
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        msg = f"Nenhum run encontrado no experimento '{experiment_name}'."
        raise RuntimeError(msg)

    return str(runs.iloc[0]["run_id"])


def recover_mlp_model(
    run_id: str | None = None,
    output_dir: str = "models/recovered",
) -> Path:
    """Recupera modelo MLP do MLflow.

    Args:
        run_id: ID do run. Se None, usa o run mais recente.
        output_dir: Diretorio para salvar o modelo recuperado.

    Returns:
        Caminho do modelo salvo.
    """
    load_dotenv_silent()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    if run_id is None:
        run_id = get_latest_run_id(DEFAULT_MLP_EXPERIMENT_NAME)
        logger.info(f"Usando run mais recente: {run_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Carregando modelo MLP de: {model_uri}")

    loaded_model = mlflow.pytorch.load_model(model_uri)
    model_path = output_path / "mlp_model.pt"
    import torch  # noqa: PLC0415

    torch.save(loaded_model.state_dict(), model_path)
    logger.info(f"Modelo MLP salvo em: {model_path}")

    return model_path


def recover_logistic_model(
    run_id: str | None = None,
    output_dir: str = "models/recovered",
) -> Path:
    """Recupera modelo Logistic Regression do MLflow.

    Args:
        run_id: ID do run. Se None, usa o run mais recente.
        output_dir: Diretorio para salvar o modelo recuperado.

    Returns:
        Caminho do modelo salvo.
    """
    load_dotenv_silent()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    if run_id is None:
        run_id = get_latest_run_id(DEFAULT_LOGISTIC_EXPERIMENT_NAME)
        logger.info(f"Usando run mais recente: {run_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Carregando modelo Logistic Regression de: {model_uri}")

    loaded_model = mlflow.sklearn.load_model(model_uri)
    model_path = output_path / "logistic_model.pkl"
    import joblib  # noqa: PLC0415

    joblib.dump(loaded_model, model_path)
    logger.info(f"Modelo Logistic Regression salvo em: {model_path}")

    return model_path


def recover_dummy_model(
    strategy: str = "most_frequent",
    run_id: str | None = None,
    output_dir: str = "models/recovered",
) -> Path:
    """Recupera modelo DummyClassifier do MLflow.

    Args:
        strategy: Estrategia do dummy classifier
            (most_frequent, stratified, uniform).
        run_id: ID do run. Se None, busca pelo run_name.
        output_dir: Diretorio para salvar o modelo recuperado.

    Returns:
        Caminho do modelo salvo.
    """
    load_dotenv_silent()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DEFAULT_DUMMY_EXPERIMENT_NAME)

    if run_id is None:
        experiment = mlflow.get_experiment_by_name(
            DEFAULT_DUMMY_EXPERIMENT_NAME
        )
        if experiment is None:
            msg = (
                f"Experimento '{DEFAULT_DUMMY_EXPERIMENT_NAME}' "
                "nao encontrado. Execute o treinamento primeiro."
            )
            raise RuntimeError(msg)

        runs: pd.DataFrame = mlflow.search_runs(  # type: ignore[assignment]
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = 'dummy_{strategy}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            msg = (
                f"Nenhum run encontrado para dummy_{strategy}. "
                "Execute o treinamento primeiro."
            )
            raise RuntimeError(msg)
        run_id = str(runs.iloc[0]["run_id"])
        logger.info(f"Usando run mais recente: {run_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Carregando modelo Dummy de: {model_uri}")

    loaded_model = mlflow.sklearn.load_model(model_uri)
    model_path = output_path / f"dummy_{strategy}_model.pkl"
    import joblib  # noqa: PLC0415

    joblib.dump(loaded_model, model_path)
    logger.info(f"Modelo Dummy salvo em: {model_path}")

    return model_path


def main() -> int:
    """Ponto de entrada para recuperacao de modelos.

    Argumentos CLI:
        --model-type: Tipo do modelo (mlp, logistic, dummy).
        --run-id: ID do run especifico (opcional).
        --strategy: Estrategia do dummy (default: most_frequent).
        --output: Diretorio para salvar o modelo.
    """
    parser = argparse.ArgumentParser(
        description="Recupera modelos treinados do MLflow"
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=_MODEL_TYPES,
        help="Tipo do modelo a recuperar",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="ID do run especifico (opcional, usa o mais recente)",
    )
    parser.add_argument(
        "--strategy",
        default="most_frequent",
        help="Estrategia do dummy classifier (default: most_frequent)",
    )
    parser.add_argument(
        "--output",
        default="models/recovered",
        help="Diretorio para salvar o modelo",
    )
    args = parser.parse_args()

    setup_logging()

    if args.model_type == "mlp":
        path = recover_mlp_model(args.run_id, args.output)
    elif args.model_type == "logistic":
        path = recover_logistic_model(args.run_id, args.output)
    elif args.model_type == "dummy":
        path = recover_dummy_model(args.strategy, args.run_id, args.output)
    else:
        msg = f"Tipo de modelo nao suportado: {args.model_type}"
        raise ValueError(msg)

    logger.info(f"Modelo recuperado com sucesso: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

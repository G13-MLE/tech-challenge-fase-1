"""Pipeline do baseline DummyClassifier para churn.

Fase 2: treino, métricas e registro no MLflow.

Este script atua como orquestrador, importando funções de outros
módulos conforme a arquitetura modular do projeto.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.logging import setup_logging
from src.constants import (
    DEFAULT_DUMMY_EXPERIMENT_NAME,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)
from src.data.load import load_telco_data
from src.data.splitting import split_train_test_stratified
from src.data.validation import (
    validate_binary_target,
    validate_required_columns,
)
from src.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from src.training import DummyTrainingConfig, run_all_strategies
from src.training.mlflow_tracking import (
    MLflowConfig,
    TrainTestData,
    build_mlflow_inputs,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


def main() -> int:
    """Ponto de entrada do script.

    Orquestra o fluxo completo:
    1. Carrega configuração e ambiente
    2. Carrega e valida dados
    3. Faz split treino/teste
    4. Treina múltiplas estratégias DummyClassifier
    5. Registra métricas no MLflow
    6. Salva resultados comparativos
    """
    # Carrega variáveis de ambiente
    load_dotenv_silent()

    # Inicializa logging estruturado
    setup_logging()

    # Configuração do pipeline
    config = DummyTrainingConfig(target_column=TARGET_COLUMN)

    # Obtém nome do experimento com prioridade
    experiment_name = get_experiment_name(
        cli_arg=None,
        env_var_name="MLFLOW_DUMMY_EXPERIMENT_NAME",
        default_name=DEFAULT_DUMMY_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(experiment_name=experiment_name)
    setup_mlflow(mlflow_config)

    # Carrega e valida dados
    df = load_telco_data()
    validate_required_columns(df, config.target_column)
    validate_binary_target(df[config.target_column], POSITIVE_LABEL)

    # Split estratificado
    X_train, X_test, y_train, y_test = split_train_test_stratified(
        df,
        config.target_column,
        config.test_size,
        config.random_seed,
    )

    # Obtem versao do dataset
    dataset_version = safe_get_dataset_version()

    # Prepara lineage de dados para MLflow
    train_test_data = TrainTestData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    train_input, test_input = build_mlflow_inputs(
        train_test_data,
        config.target_column,
        dataset_version,
    )

    # Treina todas as estrategias e obtem resultados comparativos
    results_df = run_all_strategies(
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        dataset_version,
        train_input=train_input,
        test_input=test_input,
    )

    # Salva CSV comparativo localmente (sem run adicional no MLflow)
    output_path = Path("models/dummy_baseline_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    # Exibe resumo via logging estruturado
    logger.info("Treino/aval/log no MLflow concluídos com sucesso.")
    logger.info(
        "Comparativo salvo em: %s",
        output_path,
    )
    logger.info(
        "Resultados comparativos:\n%s",
        results_df.to_string(index=False),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

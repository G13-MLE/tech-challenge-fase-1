"""Pipeline do modelo de Logistic Regression para churn."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TEST_SIZE,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.data.load import load_telco_data
from src.data.preprocessing import (
    apply_scaling,
    fit_scaler,
    mlp_preprocess_data,
)
from src.data.splitting import split_train_test_stratified
from src.data.validation import validate_required_columns
from src.pipelines.common import load_dotenv_silent
from src.training import (
    LogisticTrainingConfig,
    train_logistic_classifier,
    cross_validate_logistic,
)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treina modelo Logistic Regression para churn"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_DATASET_PATH,
        help="Caminho para o dataset de entrada",
    )
    args = parser.parse_args()

    load_dotenv_silent()

    # Carrega e valida dados utilizando as funções do módulo de dados
    df = load_telco_data(args.input)
    validate_required_columns(df, TARGET_COLUMN)

    # Preprocessing, one-hot encoding e encode target
    X, y, feature_names, _df_processed = mlp_preprocess_data(df)

    # Split treino e teste da base utilizando a função de splitting do módulo de dados
    df_for_split = pd.DataFrame(X)
    df_for_split[TARGET_COLUMN] = y
    X_train_df, X_test_df, y_train, y_test = split_train_test_stratified(
        df_for_split,
        TARGET_COLUMN,
        test_size=DEFAULT_TEST_SIZE,
        random_seed=RANDOM_SEED,
    )

   # Sem scaling: usado no CV (scaler e aplicado internamente por fold)
    X_train = X_train_df.values

    scaler = fit_scaler(X_train)
    # Com scaling: usado no treino final e avaliacao no teste
    X_train_scaled = apply_scaling(X_train, scaler)
    X_test_scaled = apply_scaling(X_test_df.values, scaler)

    # Converte target para float32
    y_train_arr: np.ndarray = np.asarray(
        y_train.values, dtype=np.float32
    )
    y_test_arr: np.ndarray = np.asarray(
        y_test.values, dtype=np.float32
    )

    # Treina e avalia
    config = LogisticTrainingConfig()

    # Validação cruzada no conjunto de treino (sem scaling - feito internamente por fold)
    cv_results = cross_validate_logistic(X_train, y_train_arr, config)

    print("[Logistic] CV Metricas:")
    for k, v in cv_results.items():
        print(f"  {k}={v:.4f}")

    result = train_logistic_classifier(
        X_train_scaled, X_test_scaled,
        y_train_arr, y_test_arr,
        config,
    )

    print("[Logistic] Treino concluido com sucesso.")
    print(
        "[Logistic] Metricas: "
        + ", ".join(
            f"{k}={v:.4f}" for k, v in result["metrics"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


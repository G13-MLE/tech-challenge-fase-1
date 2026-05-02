"""Pipeline do modelo de Logistic Regression para churn."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LOGISTIC_EXPERIMENT_NAME,
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
from src.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from src.training import (
    LogisticTrainingConfig,
    cross_validate_logistic,
    train_logistic_classifier,
)
from src.training.mlflow_tracking import MLflowConfig, setup_mlflow


@dataclass(frozen=True)
class PreparedData:
    """Dados preparados para treino e teste do modelo."""

    X_train: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split treino/teste estratificado e aplica StandardScaler."""
    df_for_split = pd.DataFrame(X)
    df_for_split[TARGET_COLUMN] = y
    X_train_df, X_test_df, y_train, y_test = split_train_test_stratified(
        df_for_split,
        TARGET_COLUMN,
        test_size=DEFAULT_TEST_SIZE,
        random_seed=RANDOM_SEED,
    )

    X_train = X_train_df.values
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaling(X_train, scaler)
    X_test_scaled = apply_scaling(X_test_df.values, scaler)

    y_train_arr: np.ndarray = np.asarray(y_train.values, dtype=np.float32)
    y_test_arr: np.ndarray = np.asarray(y_test.values, dtype=np.float32)

    return X_train, X_train_scaled, X_test_scaled, y_train_arr, y_test_arr


def _prepare_data(input_path: str) -> PreparedData:
    """Carrega, preprocessa, faz split e scaling dos dados."""
    df = load_telco_data(input_path)
    validate_required_columns(df, TARGET_COLUMN)

    X, y, feature_names, *_ = mlp_preprocess_data(df)

    X_train, X_train_scaled, X_test_scaled, y_train, y_test = _split_and_scale(
        X, y
    )

    return PreparedData(
        X_train=X_train,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(feature_names),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Treina modelo Logistic Regression para churn"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_DATASET_PATH,
        help="Caminho para o dataset de entrada",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Nome do experimento no MLflow",
    )
    args = parser.parse_args(argv)

    load_dotenv_silent()

    experiment_name = get_experiment_name(
        cli_arg=args.experiment_name,
        env_var_name="MLFLOW_LOGISTIC_EXPERIMENT_NAME",
        default_name=DEFAULT_LOGISTIC_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=experiment_name,
    )
    setup_mlflow(mlflow_config)

    prepared = _prepare_data(args.input)
    config = LogisticTrainingConfig()

    cv_results = cross_validate_logistic(
        prepared.X_train, prepared.y_train, config
    )

    dataset_version = safe_get_dataset_version()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": config.max_iter,
                "random_seed": RANDOM_SEED,
                "dataset_version": dataset_version,
                "preprocessing": "one_hot_encoding",
                "scaling": "StandardScaler",
                "scaling_fit_on": "train_only",
                "num_features": len(prepared.feature_names),
                "cv_folds": 5,
            }
        )
        mlflow.set_tag("issue", "21")
        mlflow.set_tag("baseline_family", "logistic")
        mlflow.set_tag("model_baseline", "logistic_regression")

        for k, v in cv_results.items():
            mlflow.log_metric(k, v)

        result = train_logistic_classifier(
            prepared.X_train_scaled,
            prepared.X_test_scaled,
            prepared.y_train,
            prepared.y_test,
            config,
        )

        for k, v in result["metrics"].items():
            mlflow.log_metric(f"test_{k}", v)

        coef_df = pd.DataFrame(
            {
                "feature": prepared.feature_names,
                "coefficient": result["model"].coef_[0],
            }
        ).sort_values("coefficient", key=abs, ascending=False)
        coef_path = Path("models/logistic_feature_importance.csv")
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(coef_path, index=False)
        mlflow.log_artifact(str(coef_path))

        mlflow.sklearn.log_model(result["model"], "model")

    print("[Logistic] Treino concluido com sucesso.")
    print(
        "[Logistic] Metricas teste: "
        + ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

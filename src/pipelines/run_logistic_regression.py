"""Pipeline do modelo de Logistic Regression para churn.

Utiliza sklearn Pipeline com pre-processamento completo:
imputacao, encoding, scaling e SMOTE para tratamento
de desbalanceamento.
"""

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

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _prepare_data(input_path: str) -> PreparedData:
    """Carrega, limpa, separa features/target e faz split dos dados.

    O pre-processamento (imputacao, encoding, scaling, SMOTE)
    e feito pelo sklearn Pipeline dentro das funcoes de treino.
    """
    df = load_telco_data(input_path)
    validate_required_columns(df, TARGET_COLUMN)

    if TARGET_COLUMN not in df.columns:
        msg = f"Coluna alvo '{TARGET_COLUMN}' ausente do DataFrame"
        raise ValueError(msg)

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).to_numpy(dtype=np.float64)
    X = df.drop(columns=[TARGET_COLUMN])

    feature_names = list(X.columns)

    X_with_target = X.copy()
    X_with_target["_target_"] = y
    X_train, X_test, y_train_series, y_test_series = (
        split_train_test_stratified(
            X_with_target,
            "_target_",
            test_size=DEFAULT_TEST_SIZE,
            random_seed=RANDOM_SEED,
        )
    )

    y_train: np.ndarray = y_train_series.to_numpy(dtype=np.float64)
    y_test: np.ndarray = y_test_series.to_numpy(dtype=np.float64)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
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
                "preprocessing": "sklearn_pipeline",
                "imputation": "median(numeric)_constant(categorical)",
                "encoding": "OneHotEncoder(drop=first)",
                "scaling": "StandardScaler",
                "imbalance_handling": "SMOTE",
                "original_features": len(prepared.feature_names),
                "cv_folds": 5,
            }
        )
        mlflow.set_tag("issue", "28")
        mlflow.set_tag("baseline_family", "logistic")
        mlflow.set_tag("model_baseline", "logistic_regression")

        for k, v in cv_results.items():
            mlflow.log_metric(k, v)

        result = train_logistic_classifier(
            prepared.X_train,
            prepared.X_test,
            prepared.y_train,
            prepared.y_test,
            config,
        )

        for k, v in result["metrics"].items():
            mlflow.log_metric(f"test_{k}", v)

        pipeline = result["model"]
        try:
            encoded_names = list(
                pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
        except (AttributeError, KeyError):
            encoded_names = prepared.feature_names

        mlflow.log_param("encoded_features", len(encoded_names))

        try:
            classifier = pipeline.named_steps["classifier"]
            coefs = classifier.coef_[0]
        except (AttributeError, KeyError):
            coefs = np.zeros(len(encoded_names))

        coef_df = pd.DataFrame(
            {
                "feature": encoded_names,
                "coefficient": coefs,
            }
        ).sort_values("coefficient", key=abs, ascending=False)
        coef_path = Path("models/logistic_feature_importance.csv")
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(coef_path, index=False)
        mlflow.log_artifact(str(coef_path))

        mlflow.sklearn.log_model(pipeline, "model")

    print("[Logistic] Treino concluido com sucesso.")
    print(
        "[Logistic] Metricas teste: "
        + ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

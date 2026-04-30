"""Pipeline de hyperparameter tuning para MLP com Optuna.

Executa busca de hiperparametros usando Optuna, maximizando
PR-AUC no conjunto de teste. Loga todos os trials no MLflow
para rastreabilidade.

Como usar:
    $ uv run python -m src.pipelines.run_mlp_tuning
    $ uv run python -m src.pipelines.run_mlp_tuning --n-trials 50
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch

from src.config.models import MLPConfig, TrainingConfig
from src.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MLP_EXPERIMENT_NAME,
    RANDOM_SEED,
    TARGET_COLUMN,
    THRESHOLD,
)
from src.data.load import load_telco_data
from src.data.preprocessing import (
    apply_scaling,
    fit_scaler,
    mlp_preprocess_data,
    save_scaler,
)
from src.data.splitting import split_train_test_stratified
from src.data.validation import validate_required_columns
from src.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from src.training import MLPForTraining, MLPTrainer
from src.training.metrics import compute_binary_classification_metrics
from src.training.mlflow_tracking import (
    MLflowConfig,
    TrainTestData,
    build_mlflow_inputs,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def objective(  # noqa: PLR0913, PLR0914, PLR0917
    trial: optuna.Trial,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    train_test_data: TrainTestData,
    dataset_version: str,
    args_input: str,
) -> float:
    """Funcao objetivo do Optuna para um trial.

    Args:
        trial: Trial do Optuna.
        X_train_scaled: Features de treino escaladas.
        y_train: Rótulos de treino.
        X_test_scaled: Features de teste escaladas.
        y_test: Rótulos de teste.
        feature_names: Nomes das features.
        train_test_data: Dados para MLflow lineage.
        dataset_version: Versão do dataset.
        args_input: Caminho do dataset.

    Returns:
        PR-AUC no conjunto de teste (a maximizar).
    """
    # === Espaço de busca de hiperparametros ===
    _choices: list[Any] = [(64, 32), (128, 64, 32), (256, 128, 64)]
    hidden_dims: tuple[int, ...] = trial.suggest_categorical(
        "hidden_dims", _choices
    )  # type: ignore[assignment]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    early_stopping_patience = trial.suggest_int(
        "early_stopping_patience", 5, 15, step=5
    )
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    mlp_config = MLPConfig(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )

    training_config = TrainingConfig(
        optimizer="adam",
        lr=lr,
        weight_decay=weight_decay,
        scheduler="reduce_on_plateau",
        scheduler_patience=3,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=0.001,
        batch_size=batch_size,
        max_epochs=100,
        val_split=0.2,
        random_seed=RANDOM_SEED,
    )

    with mlflow.start_run(nested=True):
        # Log de parametros
        mlflow.log_params(
            {
                "input_dim": mlp_config.input_dim,
                "hidden_dims": str(mlp_config.hidden_dims),
                "dropout_rate": mlp_config.dropout_rate,
                "use_batch_norm": mlp_config.use_batch_norm,
                "model_type": "MLP",
                "random_seed": RANDOM_SEED,
                "dataset_version": dataset_version,
                "preprocessing": "one_hot_encoding",
                "scaling": "StandardScaler",
                "scaling_fit_on": "train_only",
                "num_features": len(feature_names),
            }
        )
        mlflow.log_params(
            {
                "optimizer": training_config.optimizer,
                "lr": training_config.lr,
                "weight_decay": training_config.weight_decay,
                "batch_size": training_config.batch_size,
                "max_epochs": training_config.max_epochs,
                "early_stopping_patience": (
                    training_config.early_stopping_patience
                ),
            }
        )

        # Treinamento
        model = MLPForTraining(mlp_config)
        trainer = MLPTrainer(model, training_config)

        model_save_path = Path(f"models/trial_{trial.number}.pt")
        _history = trainer.fit(
            X_train_scaled,
            y_train,
            model_save_path=str(model_save_path),
        )
        trainer.log_to_mlflow()

        # Avaliacao no teste
        model.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(
                X_test_scaled, dtype=torch.float32
            ).to(trainer.device)
            outputs = model(X_test_tensor)
            probs = outputs["probs"].cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

        test_metrics = compute_binary_classification_metrics(
            y_true=y_test,
            y_pred=preds,
            y_proba_positive=probs,
            positive_label=None,
        )

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        # Salva modelo do trial
        mlflow.pytorch.log_model(model, "model")

        # Reporta pr_auc para o Optuna
        pr_auc = test_metrics.get("pr_auc", 0.0)
        if np.isnan(pr_auc):
            pr_auc = 0.0

        logger.info(
            f"Trial {trial.number}: pr_auc={pr_auc:.4f} "
            f"hidden_dims={hidden_dims} "
            f"lr={lr:.6f} dropout={dropout_rate:.2f}"
        )
        return float(pr_auc)


def main(  # noqa: PLR0914
    n_trials: int = 20,
    input_path: str = DEFAULT_DATASET_PATH,
    experiment_name_cli: str | None = None,
) -> None:
    """Executa o tuning completo.

    Args:
        n_trials: Numero de trials do Optuna.
        input_path: Caminho para o dataset CSV.
        experiment_name_cli: Nome do experimento no MLflow.
    """
    load_dotenv_silent()

    # Seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    experiment_name = get_experiment_name(
        cli_arg=experiment_name_cli,
        env_var_name="MLFLOW_MLP_EXPERIMENT_NAME",
        default_name=DEFAULT_MLP_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=experiment_name,
    )
    setup_mlflow(mlflow_config)

    # === Carregamento ===
    logger.info(f"Carregando dados de {input_path}")
    df = load_telco_data(input_path)
    validate_required_columns(df, TARGET_COLUMN)

    # Preprocessamento
    X, y, feature_names, _ = mlp_preprocess_data(df)
    df_for_split = pd.DataFrame(X)
    df_for_split[TARGET_COLUMN] = y
    X_train_df, X_test_df, y_train, y_test = split_train_test_stratified(
        df_for_split,
        TARGET_COLUMN,
        test_size=0.2,
        random_seed=RANDOM_SEED,
    )
    X_train = X_train_df.values
    X_test = X_test_df.values

    # Scaling
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaling(X_train, scaler)
    X_test_scaled = apply_scaling(X_test, scaler)

    y_train_arr = np.asarray(y_train.values, dtype=np.float32)
    y_test_arr = np.asarray(y_test.values, dtype=np.float32)

    train_test_data = TrainTestData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train_arr,
        y_test=y_test_arr,
    )
    dataset_version = safe_get_dataset_version()

    # MLflow lineage
    train_input, test_input = build_mlflow_inputs(
        train_test_data,
        TARGET_COLUMN,
        dataset_version,
        dataset_source_path=input_path,
    )

    # === Optuna study ===
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    with mlflow.start_run():
        mlflow.log_input(train_input, context="training")  # type: ignore[arg-type]
        mlflow.log_input(test_input, context="testing")  # type: ignore[arg-type]
        mlflow.log_param("tuning_framework", "optuna")
        mlflow.log_param("n_trials", n_trials)

        def _objective(trial: optuna.Trial) -> float:
            return objective(
                trial,
                X_train_scaled,
                y_train_arr,
                X_test_scaled,
                y_test_arr,
                feature_names,
                train_test_data,
                dataset_version,
                input_path,
            )

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

        # Log melhor trial
        best = study.best_trial
        logger.info(f"Melhor trial: {best.number} pr_auc={best.value:.4f}")
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        best_pr_auc = float(best.value) if best.value is not None else 0.0
        mlflow.log_metric("best_pr_auc", best_pr_auc)

        # Salva scaler best
        scaler_path = Path("models/scaler.pkl")
        save_scaler(scaler, str(scaler_path))
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

    # Salva resultados
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(
        reports_dir / "optuna_study.csv", index=False
    )
    logger.info(f"Estudo salvo em {reports_dir / 'optuna_study.csv'}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Tuning de hiperparametros MLP com Optuna"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Numero de trials do Optuna (default: 20)",
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        n_trials=args.n_trials,
        input_path=args.input,
        experiment_name_cli=args.experiment_name,
    )

"""Pipeline de treino para modelo MLP (Multi-Layer Perceptron).

Este script orquestra o treinamento do modelo MLP para predição de churn:
1. Carregamento dos dados brutos do dataset Telco Customer Churn
2. Pré-processamento (codificação, escalonamento SEM data leakage)
3. Divisão treino/teste estratificada
4. Configuração e treino do modelo MLP
5. Avaliação no conjunto de teste
6. Logging de métricas e modelo no MLflow

Como usar:
    $ uv run python -m src.pipelines.run_mlp
    $ uv run python -m src.pipelines.run_mlp --input path/to/data.csv
    $ uv run python -m src.pipelines.run_mlp --experiment-name churn-mlp-v2

Requerimentos:
    - Arquivo .env configurado com MLFLOW_TRACKING_URI
    - Dados na estrutura esperada (veja load_telco_data)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import torch

from src.api.logging import setup_logging
from src.config.models import MLPConfig, TrainingConfig

# Limiar para converter probabilidades em predições binárias
from src.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MLP_EXPERIMENT_NAME,
    RANDOM_SEED,
    RISK_BAND_HIGH,
    RISK_BAND_LOW,
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
    set_global_seed,
)
from src.training import MLPForTraining, MLPTrainer, cross_validate_mlp
from src.training.metrics import (
    analyze_threshold_tradeoff,
    compute_binary_classification_metrics,
    compute_calibration_metrics,
    compute_confusion_matrix,
    compute_precision_at_k,
    compute_risk_band_metrics,
)
from src.training.mlflow_tracking import (
    MLflowConfig,
    TrainTestData,
    build_mlflow_inputs,
    setup_mlflow,
)
from src.training.model_card import build_model_card
from src.training.plots import (
    save_calibration_curve,
    save_confusion_matrix_plot,
    save_loss_curve,
    save_pr_curve,
    save_roc_curve,
)

logger = logging.getLogger(__name__)


def main() -> None:  # noqa: PLR0914, PLR0915
    """função principal que executa o pipeline de treino completo.

    Orquestra todo o fluxo de ML:
    1. Parse de argumentos da linha de comando
    2. Carregamento e pré-processamento de dados
    3. Configuração do modelo e treinamento
    4. Avaliação no conjunto de teste
    5. Logging no MLflow

    Argumentos CLI:
        --input: Caminho para o dataset CSV
            (default: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)
        --experiment-name: Nome do experimento no MLflow

    Requerimentos de ambiente:
        - Arquivo .env com MLFLOW_TRACKING_URI
        - MLflow server rodando (iniciar com make docker-up)
    """
    # Configura argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Treina modelo MLP para predição de churn"
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

    # Carrega variáveis de ambiente (.env)
    load_dotenv_silent()

    # Inicializa logging estruturado
    setup_logging()

    # === SEED GLOBAL PARA REPRODUTIBILIDADE ===
    logger.info(f"Definindo seed global: {RANDOM_SEED}")
    set_global_seed(RANDOM_SEED)

    # Configura MLflow via modulo generico
    experiment_name = get_experiment_name(
        cli_arg=args.experiment_name,
        env_var_name="MLFLOW_MLP_EXPERIMENT_NAME",
        default_name=DEFAULT_MLP_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=experiment_name,
    )
    setup_mlflow(mlflow_config)

    # === 1. CARREGAMENTO DE DADOS ===
    logger.info(f"Carregando dados de {args.input}")
    df = load_telco_data(args.input)

    # Validação de dados
    validate_required_columns(df, TARGET_COLUMN)

    # === 2. PRÉ-PROCESSAMENTO (SEM SCALING AINDA) ===
    logger.info("Pré-processando dados (one-hot encoding)")
    X, y, feature_names, _ = mlp_preprocess_data(df)

    # === 3. DIVISAO TREINO/TESTE ===
    logger.info(f"Dividindo dados: treino/teste com seed={RANDOM_SEED}")
    # Cria DataFrame temporario para usar split estratificado
    df_for_split = pd.DataFrame(X)
    df_for_split[TARGET_COLUMN] = y

    X_train_df, X_test_df, y_train, y_test = split_train_test_stratified(
        df_for_split,
        TARGET_COLUMN,
        test_size=0.2,
        random_seed=RANDOM_SEED,
    )

    # Converte para numpy arrays
    X_train = X_train_df.values
    X_test = X_test_df.values

    # === 4. SCALING (APOS SPLIT - SEM DATA LEAKAGE) ===
    logger.info("Aplicando StandardScaler (fit apenas no treino)")
    # Fit scaler APENAS no treino - evita data leakage
    scaler = fit_scaler(X_train)
    # Aplica transform em treino e teste
    X_train_scaled = apply_scaling(X_train, scaler)
    X_test_scaled = apply_scaling(X_test, scaler)

    # Converte target para float32 (PyTorch)
    y_train_arr: np.ndarray = np.asarray(y_train.values, dtype=np.float32)
    y_test_arr: np.ndarray = np.asarray(y_test.values, dtype=np.float32)

    logger.info(f"Conjunto de treino: {X_train_scaled.shape[0]} amostras")
    logger.info(f"Conjunto de teste: {X_test_scaled.shape[0]} amostras")
    logger.info(f"Número de features: {X_train_scaled.shape[1]}")

    # Prepara lineage de dados para MLflow
    train_test_data = TrainTestData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train_arr,
        y_test=y_test_arr,
    )

    # Obtém versão do dataset via DVC
    dataset_version = safe_get_dataset_version()

    train_input, test_input = build_mlflow_inputs(
        train_test_data,
        TARGET_COLUMN,
        dataset_version,
        dataset_source_path=args.input,
    )

    # === 5. CONFIGURAÇÃO DO MODELO ===
    mlp_config = MLPConfig(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3,
        use_batch_norm=True,
    )

    training_config = TrainingConfig(
        optimizer="adam",
        lr=0.001,
        weight_decay=1e-5,
        scheduler="reduce_on_plateau",
        scheduler_patience=3,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        batch_size=64,
        max_epochs=100,
        val_split=0.2,
        random_seed=RANDOM_SEED,
    )

    # === 5b. CROSS-VALIDATION ===
    logger.info("Iniciando cross-validation (5 folds)")
    cv_results = cross_validate_mlp(
        X_train,
        y_train_arr,
        mlp_config,
        training_config,
        n_folds=5,
    )
    logger.info(
        "CV accuracy: %.4f (+/- %.4f)",
        cv_results["cv_accuracy_mean"],
        cv_results["cv_accuracy_std"],
    )

    # === 6. TREINAMENTO COM MLFLOW ===
    with mlflow.start_run():
        # Log inputs de dados
        mlflow.log_input(train_input, context="training")  # type: ignore[arg-type]
        mlflow.log_input(test_input, context="testing")  # type: ignore[arg-type]

        # Tags para filtragem no MLflow
        mlflow.set_tag("issue", "22")
        mlflow.set_tag("baseline_family", "mlp")
        mlflow.set_tag("model_baseline", "mlp_classifier")
        mlflow.set_tag("random_seed", str(RANDOM_SEED))

        # Registra parâmetros da arquitetura
        mlflow.log_params(
            {
                "input_dim": mlp_config.input_dim,
                "hidden_dims": str(mlp_config.hidden_dims),
                "dropout_rate": mlp_config.dropout_rate,
                "use_batch_norm": mlp_config.use_batch_norm,
                "model_type": "MLP",
                "random_seed": RANDOM_SEED,
                "dataset_version": dataset_version,
            }
        )

        # Log de pré-processamento
        mlflow.log_param("preprocessing", "one_hot_encoding")
        mlflow.log_param("scaling", "StandardScaler")
        mlflow.log_param("scaling_fit_on", "train_only")
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("cv_folds", 5)

        # Log de métricas de cross-validation
        for cv_k, cv_v in cv_results.items():
            mlflow.log_metric(cv_k, cv_v)

        # Inicializa modelo e trainer
        model = MLPForTraining(mlp_config)
        trainer = MLPTrainer(model, training_config)

        # Treina com validação e early stopping
        logger.info("Iniciando treinamento")
        model_save_path = Path("models/churn_mlp_best.pt")
        _history = trainer.fit(
            X_train_scaled, y_train_arr, model_save_path=str(model_save_path)
        )

        # Registra métricas de treino no MLflow
        trainer.log_to_mlflow()

        logger.info("Treinamento concluído")

        # === 7. AVALIAÇÃO NO CONJUNTO DE TESTE ===
        model.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(
                X_test_scaled, dtype=torch.float32
            ).to(trainer.device)
            outputs = model(X_test_tensor)
            probs = outputs["probs"].cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

        test_metrics = compute_binary_classification_metrics(
            y_true=y_test_arr,
            y_pred=preds,
            y_proba_positive=probs,
            positive_label=None,
        )
        logger.info(f"Métricas de teste: {test_metrics}")

        # --- Métricas adicionais: calibração e custo ---
        calib_metrics = compute_calibration_metrics(
            y_true=y_test_arr, y_proba_positive=probs, n_bins=10
        )
        logger.info(f"Métricas de calibração: {calib_metrics}")

        # Custo estimado: FN = LTV perdido (500), FP = campanha (50)
        cm = compute_confusion_matrix(y_true=y_test_arr, y_pred=preds)
        cost_fn = 500.0
        cost_fp = 50.0
        total_cost = (
            cm["false_negatives"] * cost_fn + cm["false_positives"] * cost_fp
        )
        logger.info(
            f"Custo estimado: R$ {total_cost:.2f} "
            f"(FN: {cm['false_negatives']} x {cost_fn}, "
            f"FP: {cm['false_positives']} x {cost_fp})"
        )

        # Precision@k e Recall@k
        pk_metrics = compute_precision_at_k(
            y_true=y_test_arr,
            y_proba_positive=probs,
            k_values=(
                100,
                250,
                500,
                int(0.05 * len(y_test_arr)),
                int(0.10 * len(y_test_arr)),
                int(0.20 * len(y_test_arr)),
            ),
        )
        logger.info(f"Precision@k/Recall@k: {pk_metrics}")

        # Bandas de risco
        risk_metrics = compute_risk_band_metrics(
            y_true=y_test_arr,
            y_proba_positive=probs,
            thresholds=(0.30, 0.60),
        )
        logger.info(f"Métricas por banda de risco: {risk_metrics}")

        # Análise de threshold tradeoff
        threshold_df = analyze_threshold_tradeoff(
            y_true=y_test_arr,
            y_proba_positive=probs,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        # Encontra threshold otimo (minimiza custo)
        optimal_idx = threshold_df["total_cost"].idxmin()
        optimal_threshold = float(
            cast("float | int", threshold_df.loc[optimal_idx, "threshold"])
        )
        optimal_total_cost = float(
            cast("float | int", threshold_df.loc[optimal_idx, "total_cost"])
        )
        logger.info(
            f"Threshold otimo (custo): {optimal_threshold} "
            f"com custo R$ {optimal_total_cost:.2f}"
        )

        # --- Salva plots como artefatos ---
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        pr_curve_path = reports_dir / "pr_curve_test.png"
        calib_curve_path = reports_dir / "calibration_curve_test.png"
        roc_curve_path = reports_dir / "roc_curve_test.png"
        cm_path = reports_dir / "confusion_matrix_test.png"
        loss_curve_path = reports_dir / "loss_curve.png"
        save_pr_curve(y_test_arr, probs, pr_curve_path)
        save_calibration_curve(y_test_arr, probs, calib_curve_path)
        save_roc_curve(y_test_arr, probs, roc_curve_path)
        save_confusion_matrix_plot(y_test_arr, preds, cm_path)
        save_loss_curve(
            trainer.history["train_loss"],
            trainer.history["val_loss"],
            loss_curve_path,
        )
        logger.info(f"Plots salvos em {reports_dir}")

        # --- Salva CSV com bandas de risco ---
        risk_df = pd.DataFrame(
            {
                "customer_id": (
                    df.iloc[y_test.index]["customerID"].values
                    if "customerID" in df.columns
                    else range(len(probs))
                ),
                "proba_churn": probs,
                "risk_band": [
                    (
                        "Low"
                        if p < RISK_BAND_LOW
                        else "Medium"
                        if p < RISK_BAND_HIGH
                        else "High"
                    )
                    for p in probs
                ],
                "true_churn": y_test_arr.astype(int),
            }
        )
        risk_csv_path = reports_dir / "risk_bands_test.csv"
        risk_df.to_csv(risk_csv_path, index=False)
        logger.info(f"Bandas de risco salvas em {risk_csv_path}")

        # --- Registra métricas no MLflow ---
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        for metric_name, metric_value in calib_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        mlflow.log_metric("test_total_cost", total_cost)
        mlflow.log_metric("test_cost_fn", cm["false_negatives"] * cost_fn)
        mlflow.log_metric("test_cost_fp", cm["false_positives"] * cost_fp)
        for metric_name, metric_value in pk_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        for metric_name, metric_value in risk_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        mlflow.log_metric("optimal_threshold_cost", optimal_threshold)
        mlflow.log_metric(
            "optimal_threshold_total_cost",
            optimal_total_cost,
        )

        # === 8. ARTEfatos no MLflow ===
        mlflow.log_artifact(str(pr_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(calib_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(roc_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(loss_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(risk_csv_path), artifact_path="reports")
        threshold_path = reports_dir / "threshold_tradeoff_test.csv"
        threshold_df.to_csv(threshold_path, index=False)
        mlflow.log_artifact(str(threshold_path), artifact_path="reports")

        # Salva modelo no MLflow registry
        mlflow.pytorch.log_model(model, "model")

        # Salva scaler para inferência
        scaler_path = Path("models/scaler.pkl")
        save_scaler(scaler, str(scaler_path))
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
        logger.info(f"Scaler salvo em {scaler_path}")

        # === 9. MODEL CARD ===
        mlp_card_values: dict[str, str | int | float] = {
            "random_seed": RANDOM_SEED,
            "dataset_version": dataset_version,
        }
        for metric_name, metric_value in test_metrics.items():
            mlp_card_values[metric_name] = metric_value
        for metric_name, metric_value in calib_metrics.items():
            mlp_card_values[metric_name] = metric_value
        mlp_card_values["total_cost"] = total_cost
        mlp_card_values["cost_fn"] = cost_fn
        mlp_card_values["cost_fp"] = cost_fp
        mlp_card_values["optimal_threshold"] = optimal_threshold
        mlp_card_values["optimal_total_cost"] = optimal_total_cost
        mlp_card_values["tn"] = cm["true_negatives"]
        mlp_card_values["fp"] = cm["false_positives"]
        mlp_card_values["fn"] = cm["false_negatives"]
        mlp_card_values["tp"] = cm["true_positives"]
        mlp_card_values.update(pk_metrics)  # type: ignore[arg-type]
        mlp_card_values.update(risk_metrics)  # type: ignore[arg-type]
        mlp_card_values.update(cv_results)  # type: ignore[arg-type]
        mlflow.log_dict(
            build_model_card("mlp", **mlp_card_values), "model_card.json"
        )
        # Salva feature names para inferência
        feature_names_path = Path("models/feature_names.json")
        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False)
        mlflow.log_artifact(
            str(feature_names_path), artifact_path="preprocessing"
        )
        logger.info(f"Feature names salvos em {feature_names_path}")

        logger.info(f"Modelo salvo em {model_save_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

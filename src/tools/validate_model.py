"""Validação periódica do modelo MLP contra baseline do MLflow.

Script de validação que re-avalia o modelo MLP atual no conjunto
de teste e compara as métricas com os baselines registrados no
MLflow. Projetado para execução semanal (cron/scheduler) conforme
o plano de monitoramento (docs/MONITORAMENTO.md Seção 2.3).

Critérios de validação:
- AUC-ROC >= 0.78 (Warning se < 0.78, Critical se < 0.72)
- F1-Score >= 0.55 (Warning se < 0.55, Critical se < 0.50)

Uso:
    uv run python -m src.tools.validate_model
    uv run python -m src.tools.validate_model --threshold-roc 0.80
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.inference import ChurnPredictor
from src.api.logging import setup_logging
from src.data.load import load_telco_data
from src.data.splitting import split_train_test_stratified
from src.pipelines.common import load_dotenv_silent
from src.training.metrics import compute_binary_classification_metrics

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/churn_mlp_best.pt")
DEFAULT_SCALER_PATH = Path("models/scaler.pkl")
DEFAULT_FEATURE_NAMES_PATH = Path("models/feature_names.json")

# Thresholds conforme MONITORAMENTO.md Seção 3.2
_DEFAULT_ROC_WARNING = 0.78
_DEFAULT_ROC_CRITICAL = 0.72
_DEFAULT_F1_WARNING = 0.55
_DEFAULT_F1_CRITICAL = 0.50
_THRESHOLD = 0.5


def _parse_args() -> argparse.Namespace:
    """Parse dos argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Validação periódica do modelo MLP de churn."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Caminho para o dataset de validação",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Caminho para o modelo MLP (.pt)",
    )
    parser.add_argument(
        "--scaler-path",
        default=str(DEFAULT_SCALER_PATH),
        help="Caminho para o scaler (.pkl)",
    )
    parser.add_argument(
        "--feature-names-path",
        default=str(DEFAULT_FEATURE_NAMES_PATH),
        help="Caminho para feature_names.json",
    )
    parser.add_argument(
        "--threshold-roc-warning",
        type=float,
        default=_DEFAULT_ROC_WARNING,
        help="Threshold AUC-ROC para warning (default: 0.78)",
    )
    parser.add_argument(
        "--threshold-roc-critical",
        type=float,
        default=_DEFAULT_ROC_CRITICAL,
        help="Threshold AUC-ROC para critical (default: 0.72)",
    )
    parser.add_argument(
        "--threshold-f1-warning",
        type=float,
        default=_DEFAULT_F1_WARNING,
        help="Threshold F1-Score para warning (default: 0.55)",
    )
    parser.add_argument(
        "--threshold-f1-critical",
        type=float,
        default=_DEFAULT_F1_CRITICAL,
        help="Threshold F1-Score para critical (default: 0.50)",
    )
    parser.add_argument(
        "--output",
        default="reports/model_validation.json",
        help="Caminho para salvar o resultado da validação",
    )
    return parser.parse_args()


def _evaluate_model_on_test_set(
    predictor: ChurnPredictor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Avalia o modelo no conjunto de teste usando ChurnPredictor.

    Para cada amostra, usa o predictor para obter a probabilidade
    de churn     e computa as métricas de classificação binária.

    Args:
        predictor: ChurnPredictor carregado com modelo e scaler.
        X_test: Features do conjunto de teste.
        y_test: Labels do conjunto de teste.

    Returns:
        Dicionário com as métricas computadas.
    """
    y_proba_list: list[float] = []
    y_pred_list: list[int] = []

    for idx in range(len(X_test)):
        row = X_test.iloc[idx]
        customer_data: dict[str, Any] = {}
        for col in X_test.columns:
            val = row[col]
            customer_data[col] = (
                int(val) if isinstance(val, (np.integer,)) else val
            )

        prob = predictor.predict(customer_data)
        y_proba_list.append(prob)
        y_pred_list.append(1 if prob >= _THRESHOLD else 0)

    y_proba = np.array(y_proba_list)
    y_pred = np.array(y_pred_list)

    # Converte y_test para binário
    if y_test.dtype == object or isinstance(y_test.iloc[0], str):
        y_true_values = y_test.values
        if isinstance(y_true_values, np.ndarray):
            y_true = (y_true_values == "Yes").astype(int)
        else:
            y_true = np.array([1 if v == "Yes" else 0 for v in y_true_values])
    else:
        y_true = y_test.values.astype(int)

    return compute_binary_classification_metrics(
        pd.Series(y_true),
        pd.Series(y_pred),
        pd.Series(y_proba),
    )


def _classify_severity(
    metrics: dict[str, float],
    roc_warning: float,
    roc_critical: float,
    f1_warning: float,
    f1_critical: float,
) -> str:
    """Classifica a severidade com base nos thresholds.

    Args:
        metrics: Métricas computadas.
        roc_warning: Threshold AUC-ROC para warning.
        roc_critical: Threshold AUC-ROC para critical.
        f1_warning: Threshold F1-Score para warning.
        f1_critical: Threshold F1-Score para critical.

    Returns:
        "OK", "WARNING" ou "CRITICAL"
    """
    roc_auc = metrics.get("roc_auc", 0.0)
    f1 = metrics.get("f1_score", 0.0)

    if roc_auc < roc_critical or f1 < f1_critical:
        return "CRITICAL"
    if roc_auc < roc_warning or f1 < f1_warning:
        return "WARNING"
    return "OK"


def validate_model(args: argparse.Namespace) -> dict[str, Any]:
    """Executa a validação periódica do modelo.

    Args:
        args: Argumentos da linha de comando.

    Returns:
        Dicionário com resultado completo da validação.
    """
    load_dotenv_silent()
    setup_logging()

    logger.info("Iniciando validação periódica do modelo...")

    # Verifica se artefatos existem
    for path, label in [
        (args.model_path, "Modelo"),
        (args.scaler_path, "Scaler"),
        (args.feature_names_path, "Feature names"),
    ]:
        if not Path(path).exists():
            msg = (
                f"{label} não encontrado: {path}. "
                "Execute 'dvc pull' para baixar os artefatos."
            )
            logger.error(msg)
            return {"status": "ERROR", "message": msg}

    # Carrega dados e split
    logger.info("Carregando dataset: %s", args.dataset_path)
    df = load_telco_data(args.dataset_path)

    # Codifica target para split
    if "Churn" in df.columns:
        df_split = df.copy()
    else:
        msg = "Coluna 'Churn' não encontrada no dataset"
        logger.error(msg)
        return {"status": "ERROR", "message": msg}

    X_train, X_test, _y_train, y_test = split_train_test_stratified(
        df_split, "Churn", test_size=0.2, random_seed=42
    )

    logger.info("Split: treino=%d, teste=%d", len(X_train), len(X_test))

    # Carrega predictor
    predictor = ChurnPredictor(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        feature_names_path=args.feature_names_path,
    )

    # Avalia modelo
    logger.info("Avaliando modelo no conjunto de teste...")
    metrics = _evaluate_model_on_test_set(predictor, X_test, y_test)

    # Classifica severidade
    severity = _classify_severity(
        metrics,
        roc_warning=args.threshold_roc_warning,
        roc_critical=args.threshold_roc_critical,
        f1_warning=args.threshold_f1_warning,
        f1_critical=args.threshold_f1_critical,
    )

    # Monta resultado
    result: dict[str, Any] = {
        "status": severity,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "thresholds": {
            "roc_auc_warning": args.threshold_roc_warning,
            "roc_auc_critical": args.threshold_roc_critical,
            "f1_warning": args.threshold_f1_warning,
            "f1_critical": args.threshold_f1_critical,
        },
        "dataset_size": len(df),
        "test_size": len(X_test),
    }

    # Salva resultado
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Log resultado
    roc_auc = metrics.get("roc_auc", 0.0)
    f1 = metrics.get("f1_score", 0.0)

    if severity == "OK":
        logger.info(
            "[OK] Validação passou. ROC-AUC=%.4f (>= %.2f), F1=%.4f (>= %.2f)",
            roc_auc,
            args.threshold_roc_warning,
            f1,
            args.threshold_f1_warning,
        )
    elif severity == "WARNING":
        logger.warning(
            "[WARN] Validação com alerta. ROC-AUC=%.4f, F1=%.4f. "
            "Agendar retreino.",
            roc_auc,
            f1,
        )
    else:
        logger.error(
            "[CRITICAL] Validação falhou. ROC-AUC=%.4f (< %.2f), "
            "F1=%.4f (< %.2f). Retreino imediato recomendado.",
            roc_auc,
            args.threshold_roc_critical,
            f1,
            args.threshold_f1_critical,
        )

    logger.info("Resultado salvo em: %s", output_path)
    return result


def main() -> None:
    """Entry point para execução via linha de comando."""
    args = _parse_args()
    result = validate_model(args)
    severity = result.get("status", "ERROR")
    if severity == "CRITICAL":
        sys.exit(2)
    if severity == "WARNING":
        sys.exit(1)
    if severity == "ERROR":
        sys.exit(3)


if __name__ == "__main__":
    main()

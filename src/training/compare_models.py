"""Modulo de comparacao entre modelos MLP e baselines.

Fornece funcoes para treinar, avaliar e comparar os modelos
DummyClassifier, LogisticRegression e MLP, gerando
metricas, analise de trade-off e visualizacoes comparativas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config.models import MLPConfig, TrainingConfig
from src.constants import RANDOM_SEED, THRESHOLD
from src.data.load import load_telco_data
from src.data.preprocessing import (
    apply_scaling,
    fit_scaler,
    mlp_preprocess_data,
)
from src.data.splitting import split_train_test_stratified
from src.training.metrics import (
    analyze_threshold_tradeoff,
    compute_binary_classification_metrics,
    compute_calibration_metrics,
    compute_confusion_matrix,
    compute_cost_analysis,
    compute_precision_at_k,
    compute_risk_band_metrics,
)
from src.training.mlp.model import MLPForTraining
from src.training.mlp.trainer import MLPTrainer

logger = logging.getLogger(__name__)

COST_FN = 500.0
COST_FP = 50.0


@dataclass(frozen=True)
class ModelResult:
    """Resultado consolidado de um modelo treinado e avaliado."""

    model_name: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    cost_analysis: dict[str, float]
    confusion_matrix_dict: dict[str, int]
    threshold_df: pd.DataFrame | None = None
    calibration: dict[str, float] | None = None


def prepare_data(  # noqa: PLR0914
    input_path: str,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Carrega e prepara dados para treino/avaliacao de modelos.

    Retorna dados prontos para uso por todos os modelos (dummy,
    logistic e MLP), garantindo mesmo split para comparacao justa.

    Args:
        input_path: Caminho para o dataset CSV.
        test_size: Proporcao do conjunto de teste.
        random_seed: Seed para reprodutibilidade.

    Returns:
        Dicionario com dados preparados e metadados.
    """
    df = load_telco_data(input_path)

    # Preprocessamento para modelos numericos
    X, y, feature_names, _ = mlp_preprocess_data(df)

    # Split estratificado
    df_for_split = pd.DataFrame(X)
    df_for_split["Churn"] = y

    X_train_df, X_test_df, y_train_s, y_test_s = split_train_test_stratified(
        df_for_split,
        "Churn",
        test_size=test_size,
        random_seed=random_seed,
    )

    X_train = X_train_df.values
    X_test = X_test_df.values

    # Scaling (fit apenas no treino => sem data leakage)
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaling(X_train, scaler)
    X_test_scaled = apply_scaling(X_test, scaler)

    y_train_arr = np.asarray(y_train_s.values, dtype=np.float32)
    y_test_arr = np.asarray(y_test_s.values, dtype=np.float32)

    return {
        "df_raw": df,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train_arr,
        "y_test": y_test_arr,
        "y_train_str": y_train_s,
        "y_test_str": y_test_s,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def train_and_evaluate_dummy(  # noqa: PLR0914
    data: dict[str, Any],
) -> dict[str, ModelResult]:
    """Treina e avalia todas as estrategias do DummyClassifier.

    Usa dados numericos (0/1) para treino, compativel com
    o preprocessamento MLP que ja codifica o target.

    Args:
        data: Dicionario com dados preparados (prepare_data).

    Returns:
        Dicionario mapeando nome da estrategia para ModelResult.
    """
    results: dict[str, ModelResult] = {}
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train = data["X_train_scaled"]
    X_test = data["X_test_scaled"]

    for strategy in ("most_frequent", "stratified", "uniform"):
        model = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # Para DummyClassifier numerico, classes_: [0.0, 1.0]
        proba = model.predict_proba(X_test)
        # Encontra indice da classe positiva (1.0)
        classes_list = list(model.classes_)
        # Classe positiva e o valor maximo (1.0 > 0.0)
        positive_idx = int(np.argmax(classes_list))
        y_proba = proba[:, positive_idx]

        y_pred_bin = (y_pred >= THRESHOLD).astype(int)
        y_true_bin = y_test.astype(int)

        metrics = compute_binary_classification_metrics(
            y_true_bin, y_pred_bin, y_proba, positive_label=None
        )

        cm = compute_confusion_matrix(y_true_bin, y_pred_bin)
        cost = compute_cost_analysis(
            y_true_bin,
            y_pred_bin,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
        )

        name = f"DummyClassifier_{strategy}"
        results[name] = ModelResult(
            model_name=name,
            metrics=metrics,
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            y_proba=y_proba,
            cost_analysis=cost,
            confusion_matrix_dict=cm,
        )

    return results


def train_and_evaluate_logistic(
    data: dict[str, Any],
    max_iter: int = 1000,
) -> dict[str, ModelResult]:
    """Treina e avalia modelo LogisticRegression.

    Args:
        data: Dicionario com dados preparados (prepare_data).
        max_iter: Maximo de iteracoes do solver.

    Returns:
        Dicionario com ModelResult para Logistic Regression.
    """
    from src.training.logistic_trainer import (  # noqa: PLC0415
        LogisticTrainingConfig,
    )

    config = LogisticTrainingConfig(max_iter=max_iter)
    model = LogisticRegression(
        max_iter=config.max_iter, random_state=RANDOM_SEED
    )
    model.fit(data["X_train_scaled"], data["y_train"])

    y_pred = model.predict(data["X_test_scaled"])
    y_proba = model.predict_proba(data["X_test_scaled"])[:, 1]
    y_true = data["y_test"]

    metrics = compute_binary_classification_metrics(
        y_true, y_pred, y_proba, positive_label=None
    )
    calib = compute_calibration_metrics(y_true, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)
    cost = compute_cost_analysis(
        y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP
    )
    threshold_df = analyze_threshold_tradeoff(
        y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP
    )
    pk = compute_precision_at_k(
        y_true,
        y_proba,
        k_values=(
            100,
            250,
            500,
            int(0.05 * len(y_true)),
            int(0.10 * len(y_true)),
            int(0.20 * len(y_true)),
        ),
    )
    # Merge precision@k metrics into main metrics
    metrics.update(pk)

    name = "LogisticRegression"
    return {
        name: ModelResult(
            model_name=name,
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            cost_analysis=cost,
            confusion_matrix_dict=cm,
            threshold_df=threshold_df,
            calibration=calib,
        )
    }


def train_and_evaluate_mlp(  # noqa: PLR0914
    data: dict[str, Any],
) -> dict[str, ModelResult]:
    """Treina e avalia modelo MLP.

    Args:
        data: Dicionario com dados preparados (prepare_data).

    Returns:
        Dicionario com ModelResult para MLP.
    """
    mlp_config = MLPConfig(
        input_dim=data["X_train_scaled"].shape[1],
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

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = MLPForTraining(mlp_config)
    trainer = MLPTrainer(model, training_config)

    model_save_path = "models/churn_mlp_comparison.pt"
    trainer.fit(
        data["X_train_scaled"],
        data["y_train"],
        model_save_path=model_save_path,
    )

    # Avaliacao no teste
    model.model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(
            data["X_test_scaled"], dtype=torch.float32
        ).to(trainer.device)
        outputs = model(X_test_tensor)
        y_proba = outputs["probs"].cpu().numpy()
        y_pred = (y_proba > THRESHOLD).astype(int)

    y_true = data["y_test"]

    metrics = compute_binary_classification_metrics(
        y_true, y_pred, y_proba, positive_label=None
    )
    calib = compute_calibration_metrics(y_true, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)
    cost = compute_cost_analysis(
        y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP
    )
    threshold_df = analyze_threshold_tradeoff(
        y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP
    )
    pk = compute_precision_at_k(
        y_true,
        y_proba,
        k_values=(
            100,
            250,
            500,
            int(0.05 * len(y_true)),
            int(0.10 * len(y_true)),
            int(0.20 * len(y_true)),
        ),
    )
    metrics.update(pk)
    risk = compute_risk_band_metrics(y_true, y_proba)
    metrics.update(risk)

    name = "MLP"
    return {
        name: ModelResult(
            model_name=name,
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            cost_analysis=cost,
            confusion_matrix_dict=cm,
            threshold_df=threshold_df,
            calibration=calib,
        )
    }


def build_comparison_table(
    all_results: dict[str, ModelResult],
) -> pd.DataFrame:
    """Constroi tabela comparativa de metricas entre modelos.

    Args:
        all_results: Dicionario mapeando nome do modelo para
            ModelResult.

    Returns:
        DataFrame com metricas de cada modelo como coluna.
    """
    primary_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]

    cost_metrics = [
        "total_cost",
        "normalized_cost",
        "cost_false_negatives",
        "cost_false_positives",
    ]

    rows: list[dict[str, Any]] = []
    for metric in primary_metrics + cost_metrics:
        row: dict[str, Any] = {"metric": metric}
        for name, result in all_results.items():
            if metric in cost_metrics:
                row[name] = result.cost_analysis.get(metric)
            else:
                row[name] = result.metrics.get(metric)
        rows.append(row)

    # Adiciona matriz de confusao
    for cm_key in (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ):
        row = {"metric": cm_key}
        for name, result in all_results.items():
            row[name] = result.confusion_matrix_dict.get(cm_key)
        rows.append(row)

    return pd.DataFrame(rows)


def build_threshold_comparison(
    all_results: dict[str, ModelResult],
) -> pd.DataFrame:
    """Constroi tabela comparativa do threshold otimo.

    Mostra o threshold que minimiza o custo total para cada
    modelo que possui analise de threshold tradeoff.

    Args:
        all_results: Dicionario mapeando nome do modelo para
            ModelResult.

    Returns:
        DataFrame com threshold otimo e custo por modelo.
    """
    rows: list[dict[str, Any]] = []
    for name, result in all_results.items():
        if result.threshold_df is not None:
            df = result.threshold_df
            optimal_idx = df["total_cost"].idxmin()
            rows.append(
                {
                    "model": name,
                    "optimal_threshold": df.loc[optimal_idx, "threshold"],
                    "optimal_total_cost": df.loc[optimal_idx, "total_cost"],
                    "optimal_precision": df.loc[optimal_idx, "precision"],
                    "optimal_recall": df.loc[optimal_idx, "recall"],
                    "optimal_f1_score": df.loc[optimal_idx, "f1_score"],
                    "false_negatives": df.loc[optimal_idx, "false_negatives"],
                    "false_positives": df.loc[optimal_idx, "false_positives"],
                }
            )

    return pd.DataFrame(rows)


def plot_roc_comparison(
    all_results: dict[str, ModelResult],
    output_path: str | Path,
) -> None:
    """Gera curva ROC comparativa entre modelos.

    Args:
        all_results: Dicionario de nome -> ModelResult.
        output_path: Caminho para salvar a imagem PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, result in all_results.items():
        fpr, tpr, _ = roc_curve(result.y_true, result.y_proba)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Comparacao de Modelos")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_comparison(
    all_results: dict[str, ModelResult],
    output_path: str | Path,
) -> None:
    """Gera curva Precision-Recall comparativa.

    Args:
        all_results: Dicionario de nome -> ModelResult.
        output_path: Caminho para salvar a imagem PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, result in all_results.items():
        prec_vals, rec_vals, _ = precision_recall_curve(
            result.y_true, result.y_proba
        )
        ax.plot(rec_vals, prec_vals, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Comparacao de Modelos")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    all_results: dict[str, ModelResult],
    output_path: str | Path,
) -> None:
    """Gera subplots com matrizes de confusao comparativas.

    Args:
        all_results: Dicionario de nome -> ModelResult.
        output_path: Caminho para salvar a imagem PNG.
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    labels = ["No Churn", "Churn"]
    for ax, (name, result) in zip(axes, all_results.items()):
        cm = confusion_matrix(result.y_true, result.y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(name)

    fig.suptitle("Confusion Matrix - Comparacao", fontsize=14)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Gera grafico de barras comparando custo total por modelo.

    Args:
        comparison_df: DataFrame da build_comparison_table.
        output_path: Caminho para salvar a imagem PNG.
    """
    cost_row = comparison_df[comparison_df["metric"] == "total_cost"]
    if cost_row.empty:
        return

    models = [c for c in cost_row.columns if c != "metric"]
    costs = [float(cost_row[c].values[0]) for c in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, costs, color=["#cccccc", "#4c72b0", "#dd8452"])
    ax.set_ylabel("Custo Total (R$)")
    ax.set_title("Custo Total Estimado por Modelo")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.01,
            f"R$ {cost:,.0f}",
            ha="center",
            va="bottom",
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(
    all_results: dict[str, ModelResult],
    output_path: str | Path,
) -> None:
    """Gera grafico de trade-off precision/recall/custo.

    Inclui apenas modelos com threshold_df disponivel.

    Args:
        all_results: Dicionario de nome -> ModelResult.
        output_path: Caminho para salvar a imagem PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"LogisticRegression": "#4c72b0", "MLP": "#dd8452"}
    linestyles = {"LogisticRegression": "-", "MLP": "--"}

    for name, result in all_results.items():
        if result.threshold_df is None:
            continue
        df = result.threshold_df
        color = colors.get(name, "#333333")
        ls = linestyles.get(name, "-")

        axes[0].plot(
            df["threshold"],
            df["precision"],
            label=name,
            color=color,
            linestyle=ls,
        )
        axes[1].plot(
            df["threshold"],
            df["recall"],
            label=name,
            color=color,
            linestyle=ls,
        )
        axes[2].plot(
            df["threshold"],
            df["total_cost"],
            label=name,
            color=color,
            linestyle=ls,
        )

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision vs Threshold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall vs Threshold")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Custo Total (R$)")
    axes[2].set_title("Custo Total vs Threshold")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Trade-off: Precision, Recall e Custo", fontsize=14)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_radar(
    all_results: dict[str, ModelResult],
    output_path: str | Path,
) -> None:
    """Gera grafico radar comparativo das metricas principais.

    Args:
        all_results: Dicionario de nome -> ModelResult.
        output_path: Caminho para salvar a imagem PNG.
    """
    metrics_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]
    labels_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    # Filtra modelos para comparacao justa
    compare_results = {
        k: v
        for k, v in all_results.items()
        if "Dummy" not in k or k == "DummyClassifier_stratified"
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    colors = [
        "#cccccc",
        "#4c72b0",
        "#dd8452",
        "#55a868",
    ]

    for idx, (name, result) in enumerate(compare_results.items()):
        if idx >= len(colors):
            break
        values = [result.metrics.get(m, 0.0) for m in metrics_keys]
        values_arr = np.concatenate((values, [values[0]]))
        ax.plot(
            angles,
            values_arr,
            "o-",
            linewidth=2,
            label=name,
            color=colors[idx],
        )
        ax.fill(
            angles,
            values_arr,
            alpha=0.1,
            color=colors[idx],
        )

    ax.set_thetagrids(  # type: ignore[attr-defined]
        angles[:-1] * 180 / np.pi,
        [labels_map[m] for m in metrics_keys],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Comparacao de Metricas por Modelo", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_metrics_table(
    all_results: dict[str, ModelResult],
    comparison_df: pd.DataFrame,
) -> list[str]:
    """Gera linhas de tabela markdown com metricas."""
    lines: list[str] = []
    lines.append("| Metrica | " + " | ".join(comparison_df.columns[1:]) + " |")
    lines.append(
        "|---------|"
        + "|".join(["-------"] * (len(comparison_df.columns) - 1))
        + "|"
    )
    for _, row in comparison_df.iterrows():
        vals = []
        for c in comparison_df.columns[1:]:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {row['metric']} | " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def _render_tradeoff_section(
    all_results: dict[str, ModelResult],
) -> list[str]:
    """Gera secao de trade-off FN/FP do relatorio."""
    lines: list[str] = []
    lines.append("## 2. Analise de Trade-off FN/FP (Custo de Negocio)")
    lines.append("")
    lines.append(
        "Em telecom, o custo de um **False Negative** "
        "(nao detectar um churner) e tipicamente muito "
        "maior que o custo de um **False Positive** "
        "(oferecer retencao para um cliente leal)."
    )
    lines.append("")
    lines.append(
        f"- **Custo FN (LTV perdido por churner nao detectado):** "
        f"R$ {COST_FN:,.0f}"
    )
    lines.append(
        f"- **Custo FP (campanha de retencao desnecessaria):** "
        f"R$ {COST_FP:,.0f}"
    )
    lines.append("")

    for name, result in all_results.items():
        fn = result.confusion_matrix_dict.get("false_negatives", 0)
        fp = result.confusion_matrix_dict.get("false_positives", 0)
        total = result.cost_analysis.get("total_cost", 0.0)
        lines.append(
            f"- **{name}:** FN={fn}, FP={fp}, Custo total=R$ {total:,.0f}"
        )
    lines.append("")
    return lines


def _render_threshold_section(
    threshold_df: pd.DataFrame,
) -> list[str]:
    """Gera secao de threshold otimo do relatorio."""
    lines: list[str] = []
    if threshold_df.empty:
        return lines

    lines.append("## 3. Threshold Otimo por Modelo")
    lines.append("")
    lines.append(
        "O threshold otimo e definido como o valor que "
        "minimiza o custo total de negocio."
    )
    lines.append("")
    cols = threshold_df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["-------"] * len(cols)) + "|")
    for _, row in threshold_df.iterrows():
        vals = [
            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
            for c in cols
        ]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def _render_cm_section(
    all_results: dict[str, ModelResult],
) -> list[str]:
    """Gera secao de matrizes de confusao do relatorio."""
    lines: list[str] = []
    lines.append("## 4. Matrizes de Confusao")
    lines.append("")
    lines.append(
        "| Modelo | TN (No Churn correto) | FP (Falso alarme) | "
        "FN (Churner nao detectado) | TP (Churner detectado) |"
    )
    lines.append(
        "|--------|----------------------|-------------------|"
        "----------------------------|----------------------|"
    )
    for name, result in all_results.items():
        cm = result.confusion_matrix_dict
        lines.append(
            f"| {name} | {cm.get('true_negatives', 0)} | "
            f"{cm.get('false_positives', 0)} | "
            f"{cm.get('false_negatives', 0)} | "
            f"{cm.get('true_positives', 0)} |"
        )
    lines.append("")
    return lines


def _render_calibration_section(
    all_results: dict[str, ModelResult],
) -> list[str]:
    """Gera secao de calibracao do relatorio."""
    lines: list[str] = []
    lines.append("## 5. Calibracao das Probabilidades")
    lines.append("")
    for name, result in all_results.items():
        if result.calibration is not None:
            brier = result.calibration.get("brier_score", float("nan"))
            ece = result.calibration.get(
                "expected_calibration_error",
                float("nan"),
            )
            lines.append(
                f"- **{name}:** Brier Score = {brier:.4f}, ECE = {ece:.4f}"
            )
    lines.append("")
    return lines


def _render_conclusions(  # noqa: PLR0915
    all_results: dict[str, ModelResult],
) -> list[str]:
    """Gera secao de conclusoes do relatorio."""
    lines: list[str] = []
    lines.append("## 6. Conclusoes")
    lines.append("")

    best_auc = max(
        all_results.items(),
        key=lambda x: x[1].metrics.get("roc_auc", 0.0),
    )
    best_f1 = max(
        all_results.items(),
        key=lambda x: x[1].metrics.get("f1_score", 0.0),
    )
    best_cost = min(
        all_results.items(),
        key=lambda x: x[1].cost_analysis.get("total_cost", float("inf")),
    )

    lines.append(f"### Melhor Modelo por ROC-AUC: **{best_auc[0]}**")
    lines.append(f"- ROC-AUC = {best_auc[1].metrics.get('roc_auc', 0.0):.4f}")
    lines.append("")

    lines.append(f"### Melhor Modelo por F1-Score: **{best_f1[0]}**")
    lines.append(f"- F1-Score = {best_f1[1].metrics.get('f1_score', 0.0):.4f}")
    lines.append("")

    lines.append(f"### Menor Custo de Negocio: **{best_cost[0]}**")
    lines.append(
        f"- Custo total = R$ "
        f"{best_cost[1].cost_analysis.get('total_cost', 0.0):,.0f}"
    )
    lines.append("")

    mlp_result = all_results.get("MLP")
    logistic_result = all_results.get("LogisticRegression")
    dummy_strat = all_results.get("DummyClassifier_stratified")

    if mlp_result and logistic_result:
        mlp_auc = mlp_result.metrics.get("roc_auc", 0.0)
        log_auc = logistic_result.metrics.get("roc_auc", 0.0)
        mlp_f1 = mlp_result.metrics.get("f1_score", 0.0)
        log_f1 = logistic_result.metrics.get("f1_score", 0.0)

        lines.append("### MLP vs Logistic Regression")
        lines.append("")
        if mlp_auc > log_auc:
            lines.append(
                f"O MLP superou a Logistic Regression em "
                f"ROC-AUC ({mlp_auc:.4f} vs {log_auc:.4f}), "
                f"indicando melhor capacidade discriminativa."
            )
        else:
            lines.append(
                f"A Logistic Regression superou o MLP em "
                f"ROC-AUC ({log_auc:.4f} vs {mlp_auc:.4f}), "
                f"indicando que o modelo linear e suficiente."
            )
        lines.append("")

        if mlp_f1 > log_f1:
            lines.append(
                f"O MLP obteve F1-Score superior "
                f"({mlp_f1:.4f} vs {log_f1:.4f})."
            )
        else:
            lines.append(
                f"A Logistic Regression obteve F1-Score "
                f"superior ({log_f1:.4f} vs {mlp_f1:.4f})."
            )
        lines.append("")

    if dummy_strat and mlp_result:
        dummy_auc = dummy_strat.metrics.get("roc_auc", 0.0)
        mlp_auc = mlp_result.metrics.get("roc_auc", 0.0)
        lines.append("### MLP vs Dummy Baseline")
        lines.append("")
        lines.append(
            f"O MLP apresenta ROC-AUC de {mlp_auc:.4f} contra "
            f"{dummy_auc:.4f} do DummyClassifier (stratified), "
            f"confirmando ganho significativo sobre o baseline."
        )
        lines.append("")

    lines.append("### Recomendacao")
    lines.append("")
    if mlp_result and logistic_result:
        mlp_auc = mlp_result.metrics.get("roc_auc", 0.0)
        log_auc = logistic_result.metrics.get("roc_auc", 0.0)
        mlp_cost = mlp_result.cost_analysis.get("total_cost", float("inf"))
        log_cost = logistic_result.cost_analysis.get(
            "total_cost", float("inf")
        )

        if mlp_auc > log_auc and mlp_cost < log_cost:
            lines.append(
                "**Recomendacao: MLP** - Superior em ROC-AUC "
                "e com menor custo de negocio. E o modelo "
                "preferido para producao."
            )
        elif log_auc >= mlp_auc and log_cost <= mlp_cost:
            lines.append(
                "**Recomendacao: Logistic Regression** - Com "
                "desempenho igual ou superior ao MLP e menor "
                "complexidade. E o modelo preferido para producao."
            )
        else:
            lines.append(
                "**Recomendacao: Analise case-by-case** - Os "
                "modelos tem trade-offs diferentes. Considere "
                "o custo de implantacao e a necessidade de "
                "explicabilidade."
            )
    else:
        lines.append("Dados insuficientes para recomendacao definitiva.")
    lines.append("")
    return lines


def generate_markdown_report(
    all_results: dict[str, ModelResult],
    comparison_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> str:
    """Gera relatorio markdown com analise comparativa completa.

    Args:
        all_results: Dicionario de nome -> ModelResult.
        comparison_df: DataFrame de build_comparison_table.
        threshold_df: DataFrame de build_threshold_comparison.

    Returns:
        String com conteudo markdown do relatorio.
    """
    lines: list[str] = []

    lines.append(
        "# Comparacao: MLP vs Modelos Baseline para Predicao de Churn"
    )
    lines.append("")
    lines.append(
        "Este documento apresenta a comparacao completa entre "
        "o modelo MLP e os modelos baseline (DummyClassifier "
        "e Logistic Regression) para predicao de churn."
    )
    lines.append("")

    # 1. Metricas
    lines.append("## 1. Tabela Comparativa de Metricas")
    lines.append("")
    lines.extend(_render_metrics_table(all_results, comparison_df))

    # 2. Trade-off
    lines.extend(_render_tradeoff_section(all_results))

    # 3. Threshold
    lines.extend(_render_threshold_section(threshold_df))

    # 4. Matrizes de confusao
    lines.extend(_render_cm_section(all_results))

    # 5. Calibracao
    lines.extend(_render_calibration_section(all_results))

    # 6. Conclusoes
    lines.extend(_render_conclusions(all_results))

    # 7. Visualizacoes
    lines.append("## 7. Visualizacoes")
    lines.append("")
    lines.append("As seguintes visualizacoes foram geradas em `reports/`:")
    lines.append("")
    lines.append("- `comparison_roc_curve.png`: Curva ROC comparativa")
    lines.append(
        "- `comparison_pr_curve.png`: Curva Precision-Recall comparativa"
    )
    lines.append("- `comparison_confusion_matrices.png`: Matrizes de confusao")
    lines.append("- `comparison_cost.png`: Custo total por modelo")
    lines.append(
        "- `comparison_threshold_tradeoff.png`: "
        "Trade-off precision/recall/custo por threshold"
    )
    lines.append("- `comparison_metrics_radar.png`: Radar de metricas")
    lines.append("")

    return "\n".join(lines)

"""Geradores de plots para avaliacao de modelos de churn.

Fornece funcoes padronizadas para salvar curvas ROC, PR,
calibracao, matriz de confusao e curvas de loss
como artefatos do MLflow. Usa matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def save_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    filepath: str | Path,
) -> None:
    """Salva curva Precision-Recall em arquivo.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_proba: Probabilidades preditas para a classe positiva.
        filepath: Caminho para salvar a imagem (PNG).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, marker="", label="PR Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    filepath: str | Path,
    n_bins: int = 10,
) -> None:
    """Salva curva de calibracao em arquivo.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_proba: Probabilidades preditas para a classe positiva.
        filepath: Caminho para salvar a imagem (PNG).
        n_bins: Numero de bins para a curva de calibracao.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Modelo")
    ax.plot([0, 1], [0, 1], "k--", label="Perfeitamente calibrado")
    ax.set_xlabel("Probabilidade predita media")
    ax.set_ylabel("Fracao de positivos")
    ax.set_title("Curva de Calibracao")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    filepath: str | Path,
) -> None:
    """Salva curva ROC em arquivo.

    Args:
        y_true: Rotulos verdadeiros (0/1).
        y_proba: Probabilidades preditas para a classe positiva.
        filepath: Caminho para salvar a imagem (PNG).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, marker="", label="ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: str | Path,
    labels: list[str] | None = None,
) -> None:
    """Salva matriz de confusao como imagem em arquivo.

    Args:
        y_true: Rotulos verdadeiros (0/1).
        y_pred: Predicoes do modelo (0/1).
        filepath: Caminho para salvar a imagem (PNG).
        labels: Nomes das classes para os eixos.
            Default: ["No Churn", "Churn"].
    """
    if labels is None:
        labels = ["No Churn", "Churn"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    filepath: str | Path,
) -> None:
    """Salva curva de loss (treino e validacao) em arquivo.

    Args:
        train_losses: Lista de losses de treino por epoca.
        val_losses: Lista de losses de validacao por epoca.
        filepath: Caminho para salvar a imagem (PNG).
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="", label="Train Loss")
    ax.plot(epochs, val_losses, marker="", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

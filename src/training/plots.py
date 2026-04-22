"""Geradores de plots para avaliacao de modelos de churn.

Fornece funcoes padronizadas para salvar curvas PR e calibracao
como artefatos do MLflow. Usa matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


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

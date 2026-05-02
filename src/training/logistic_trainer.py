"""Modulo para treinar um modelo de Logistic Regression.

Fornece funcoes para treinar e avaliar o modelo de
Logistic Regression com tracking no MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.constants import RANDOM_SEED
from src.data.preprocessing import apply_scaling, fit_scaler
from src.training.metrics import compute_binary_classification_metrics


@dataclass(frozen=True)
class LogisticTrainingConfig:
    """Configuração para treino do LogisticRegression."""

    max_iter: int = 1000
    random_seed: int = RANDOM_SEED


def train_logistic_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: LogisticTrainingConfig,
) -> dict[str, Any]:
    """Treina um modelo de Logistic Regression e avalia seu desempenho."""

    model = LogisticRegression(
        max_iter=config.max_iter, random_state=config.random_seed
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_binary_classification_metrics(
        y_test, y_pred, y_proba, positive_label=None
    )
    return {"model": model, "metrics": metrics}


def cross_validate_logistic(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    config: LogisticTrainingConfig,
    n_folds: int = 5,
) -> dict[str, float]:
    """Realiza cross-validation com Logistic Regression
    e retorna métricas médias e std.
    O cross-validation deve ser realizado no conjunto
    de treino antes de aplicar scaling, para evitar data leakage.
    Depois de definir os folds, o scaling deve ser aplicado
    dentro de cada fold (fit no treino do fold, apply no teste do fold).
    """

    # Retorna: cv_accuracy_mean, cv_accuracy_std, cv_f1_mean, cv_f1_std...
    cv = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config.random_seed
    )
    cv_results = []
    for train_idx, test_idx in cv.split(X_train_raw, y_train):
        X_fold_train = X_train_raw[train_idx]
        X_fold_val = X_train_raw[test_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[test_idx]

        # Scaling dentro do fold - evita data leakage
        fold_scaler = fit_scaler(X_fold_train)
        X_fold_train_scaled = apply_scaling(X_fold_train, fold_scaler)
        X_fold_val_scaled = apply_scaling(X_fold_val, fold_scaler)

        model = LogisticRegression(
            max_iter=config.max_iter, random_state=config.random_seed
        )
        model.fit(X_fold_train_scaled, y_fold_train)
        y_pred = model.predict(X_fold_val_scaled)
        y_proba = model.predict_proba(X_fold_val_scaled)[:, 1]
        metrics = compute_binary_classification_metrics(
            y_fold_val, y_pred, y_proba, positive_label=None
        )
        cv_results.append(metrics)

    return {
        "cv_accuracy_mean": float(
            np.mean([r["accuracy"] for r in cv_results])
        ),
        "cv_accuracy_std": float(np.std([r["accuracy"] for r in cv_results])),
        "cv_precision_mean": float(
            np.mean([r["precision"] for r in cv_results])
        ),
        "cv_precision_std": float(
            np.std([r["precision"] for r in cv_results])
        ),
        "cv_recall_mean": float(np.mean([r["recall"] for r in cv_results])),
        "cv_recall_std": float(np.std([r["recall"] for r in cv_results])),
        "cv_f1_mean": float(np.mean([r["f1_score"] for r in cv_results])),
        "cv_f1_std": float(np.std([r["f1_score"] for r in cv_results])),
        "cv_roc_auc_mean": float(np.mean([r["roc_auc"] for r in cv_results])),
        "cv_roc_auc_std": float(np.std([r["roc_auc"] for r in cv_results])),
    }

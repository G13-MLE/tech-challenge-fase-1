"""Módulo para treinar um modelo de Logistic Regression.

Fornece funções para treinar e avaliar o modelo de
Logistic Regression com tracking no MLflow.
Utiliza sklearn Pipeline para reprodutibilidade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.constants import RANDOM_SEED
from src.features.pipeline import build_logistic_pipeline
from src.training.metrics import compute_binary_classification_metrics


@dataclass(frozen=True)
class LogisticTrainingConfig:
    """Configuração para treino do LogisticRegression."""

    max_iter: int = 1000
    random_seed: int = RANDOM_SEED


def train_logistic_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: LogisticTrainingConfig,
) -> dict[str, Any]:
    """Treina pipeline sklearn com Logistic Regression e avalia desempenho.

    Constrói um pipeline com pré-processamento completo:
    imputação, encoding, scaling, SMOTE e classificador.
    """

    pipeline = build_logistic_pipeline(
        max_iter=config.max_iter,
        random_seed=config.random_seed,
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_binary_classification_metrics(
        y_test, y_pred, y_proba, positive_label=None
    )
    return {"model": pipeline, "metrics": metrics}


def cross_validate_logistic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: LogisticTrainingConfig,
    n_folds: int = 5,
) -> dict[str, float]:
    """Realiza cross-validation com Logistic Regression via sklearn Pipeline.

    Constrói um pipeline novo por fold para evitar data leakage.
    O pipeline gerencia internamente scaling e SMOTE dentro de cada fold.
    """

    cv = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config.random_seed
    )
    cv_results = []
    for train_idx, test_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[test_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[test_idx]

        pipeline = build_logistic_pipeline(
            max_iter=config.max_iter,
            random_seed=config.random_seed,
        )
        pipeline.fit(X_fold_train, y_fold_train)
        y_pred = pipeline.predict(X_fold_val)
        y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
        metrics = compute_binary_classification_metrics(
            y_fold_val, y_pred, y_proba, positive_label=None
        )
        cv_results.append(metrics)

    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    aggregated: dict[str, float] = {}
    for key in metric_keys:
        values = [r[key] for r in cv_results]
        aggregated[f"cv_{key}_mean"] = float(np.mean(values))
        aggregated[f"cv_{key}_std"] = float(np.std(values))
    return aggregated

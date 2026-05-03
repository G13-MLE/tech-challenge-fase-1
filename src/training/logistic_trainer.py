"""Modulo para treinar um modelo de Logistic Regression.

Fornece funcoes para treinar e avaliar o modelo de
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
    """Configuracao para treino do LogisticRegression."""

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

    Constroi um pipeline com pre-processamento completo:
    imputacao, encoding, scaling, SMOTE e classificador.
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

    Constroi um pipeline novo por fold para evitar data leakage.
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

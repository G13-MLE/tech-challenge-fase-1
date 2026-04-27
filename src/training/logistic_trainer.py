"""Módulo para treinar um modelo de Logistic Regression.
Este módulo fornece funções para treinar e avaliar o modelo de Logistic Regression com tracking no MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.constants import RANDOM_SEED, TARGET_COLUMN, DEFAULT_TEST_SIZE
from src.training.metrics import compute_binary_classification_metrics


@dataclass(frozen=True)
class LogisticTrainingConfig:
    """Configuração para treino do LogisticRegression."""
    
    max_iter: int = 1000
    test_size: float = DEFAULT_TEST_SIZE
    random_seed: int = RANDOM_SEED
    target_column: str = TARGET_COLUMN


def train_logistic_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: LogisticTrainingConfig,
) -> dict[str, Any]:
    """Treina um modelo de Logistic Regression e avalia seu desempenho."""
    
    model = LogisticRegression(max_iter=config.max_iter, random_state=config.random_seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_binary_classification_metrics(
        y_test, y_pred, y_proba, positive_label=None)
    return {"model": model, "metrics": metrics}
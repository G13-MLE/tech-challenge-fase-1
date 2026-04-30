"""Testes para o módulo logistic_trainer."""

from __future__ import annotations

import numpy as np

from src.training import LogisticTrainingConfig, train_logistic_classifier
from src.training.logistic_trainer import cross_validate_logistic

EXPECTED_METRICS_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
}

EXPECTED_CV_KEYS = {
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_precision_mean",
    "cv_precision_std",
    "cv_recall_mean",
    "cv_recall_std",
    "cv_f1_mean",
    "cv_f1_std",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
}


def make_numeric_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Cria dataset numerico minimo para testes."""
    rng = np.random.default_rng(42)
    X = rng.random((100, 10)).astype(np.float32)
    y = np.array([0, 1] * 50, dtype=np.float32)
    return X[:80], X[80:], y[:80], y[80:]


def test_train_logistic_classifier_returns_expected_keys() -> None:
    """train_logistic_classifier deve retornar dict com model e metrics."""
    X_train, X_test, y_train, y_test = make_numeric_data()
    config = LogisticTrainingConfig(random_seed=42)

    result = train_logistic_classifier(
        X_train, X_test, y_train, y_test, config
    )

    assert "model" in result
    assert "metrics" in result


def test_train_logistic_classifier_metrics_keys() -> None:
    """Metricas devem conter todas as chaves esperadas."""
    X_train, X_test, y_train, y_test = make_numeric_data()
    config = LogisticTrainingConfig(random_seed=42)

    result = train_logistic_classifier(
        X_train, X_test, y_train, y_test, config
    )

    assert set(result["metrics"].keys()) == EXPECTED_METRICS_KEYS


def test_train_logistic_classifier_model_is_fitted() -> None:
    """Modelo deve ter coef_ apos o treino."""
    X_train, X_test, y_train, y_test = make_numeric_data()
    config = LogisticTrainingConfig(random_seed=42)

    result = train_logistic_classifier(
        X_train, X_test, y_train, y_test, config
    )

    assert hasattr(result["model"], "coef_")


def test_cross_validate_logistic_returns_expected_keys() -> None:
    """cross_validate_logistic deve retornar todas as chaves esperadas."""
    X_train, _, y_train, _ = make_numeric_data()
    config = LogisticTrainingConfig(random_seed=42)

    cv_results = cross_validate_logistic(X_train, y_train, config)

    assert set(cv_results.keys()) == EXPECTED_CV_KEYS


def test_cross_validate_logistic_values_in_range() -> None:
    """Metricas do CV devem estar entre 0 e 1."""
    X_train, _, y_train, _ = make_numeric_data()
    config = LogisticTrainingConfig(random_seed=42)

    cv_results = cross_validate_logistic(X_train, y_train, config)

    for k, v in cv_results.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} fora do intervalo [0, 1]"

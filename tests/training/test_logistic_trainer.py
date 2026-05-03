"""Testes para o módulo logistic_trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training import LogisticTrainingConfig, train_logistic_classifier
from src.training.logistic_trainer import cross_validate_logistic

EXPECTED_METRICS_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "brier_score",
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


def make_mixed_df(n_samples: int = 100) -> pd.DataFrame:
    """Cria DataFrame sintetico com colunas numericas e categoricas."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n_samples),
            "MonthlyCharges": rng.uniform(20, 120, n_samples),
            "TotalCharges": rng.uniform(0, 8000, n_samples),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n_samples
            ),
            "gender": rng.choice(["Female", "Male"], n_samples),
        }
    )


def make_binary_target(n_samples: int = 100) -> np.ndarray:
    """Cria target binario com ~30% de positivos."""
    y = np.zeros(n_samples, dtype=np.float64)
    y[: n_samples // 3] = 1.0
    rng = np.random.default_rng(42)
    rng.shuffle(y)
    return y


def make_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Cria dados sinteticos completos (DataFrames + arrays)."""
    X = make_mixed_df(100)
    y = make_binary_target(100)
    split_idx = 80
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y[:split_idx],
        y[split_idx:],
    )


def test_train_logistic_classifier_returns_expected_keys() -> None:
    """train_logistic_classifier deve retornar dict com model e metrics."""
    X_train, X_test, y_train, y_test = make_data()
    config = LogisticTrainingConfig(random_seed=42)

    result = train_logistic_classifier(
        X_train, X_test, y_train, y_test, config
    )

    assert "model" in result
    assert "metrics" in result


def test_train_logistic_classifier_metrics_keys() -> None:
    """Metricas devem conter todas as chaves esperadas."""
    X_train, X_test, y_train, y_test = make_data()
    config = LogisticTrainingConfig(random_seed=42)

    result = train_logistic_classifier(
        X_train, X_test, y_train, y_test, config
    )

    assert set(result["metrics"].keys()) == EXPECTED_METRICS_KEYS


def test_train_logistic_classifier_model_is_fitted() -> None:
    """Pipeline deve estar fitado apos o treino."""
    X_train, X_test, y_train, y_test = make_data()
    config = LogisticTrainingConfig(random_seed=42)

    result = train_logistic_classifier(
        X_train, X_test, y_train, y_test, config
    )

    pipeline = result["model"]
    classifier = pipeline.named_steps["classifier"]
    assert hasattr(classifier, "coef_")


def test_cross_validate_logistic_returns_expected_keys() -> None:
    """cross_validate_logistic deve retornar todas as chaves esperadas."""
    X_train, _, y_train, _ = make_data()
    config = LogisticTrainingConfig(random_seed=42)

    cv_results = cross_validate_logistic(X_train, y_train, config)

    assert set(cv_results.keys()) == EXPECTED_CV_KEYS


def test_cross_validate_logistic_values_in_range() -> None:
    """Metricas do CV devem estar entre 0 e 1."""
    X_train, _, y_train, _ = make_data()
    config = LogisticTrainingConfig(random_seed=42)

    cv_results = cross_validate_logistic(X_train, y_train, config)

    for k, v in cv_results.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} fora do intervalo [0, 1]"

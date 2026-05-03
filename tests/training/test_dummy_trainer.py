"""Testes para o módulo dummy_trainer.

Testa as funções de treinamento e avaliação do DummyClassifier.
"""

from __future__ import annotations

from typing import Self

import pandas as pd

from src.training import DummyTrainingConfig, train_dummy_strategy
from src.training.dummy_trainer import run_all_strategies

POSITIVE_LABEL = "Yes"
EXPECTED_STRATEGIES_COUNT = 3


def make_dummy_df() -> pd.DataFrame:
    """Cria dataset mínimo para testes."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 1],
            "Churn": ["No", "Yes", "No", "Yes", "No", "Yes"],
        }
    )


def test_train_dummy_strategy_returns_expected_keys(
    monkeypatch: object,
) -> None:
    """train_dummy_strategy deve retornar dicionário com chaves esperadas."""
    df = make_dummy_df()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split simples
    X_train, X_test = X.iloc[:3], X.iloc[3:]
    y_train, y_test = y.iloc[:3], y.iloc[3:]

    config = DummyTrainingConfig(random_seed=42)

    result = train_dummy_strategy(
        X_train, X_test, y_train, y_test, "most_frequent", config
    )

    assert "strategy" in result
    assert "model" in result
    assert "metrics" in result
    assert result["strategy"] == "most_frequent"
    assert "accuracy" in result["metrics"]
    assert "f1_score" in result["metrics"]


def test_train_dummy_strategy_different_strategies() -> None:
    """Diferentes estratégias devem produzir resultados diferentes."""
    df = make_dummy_df()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test = X.iloc[:3], X.iloc[3:]
    y_train, y_test = y.iloc[:3], y.iloc[3:]

    config = DummyTrainingConfig(random_seed=42)

    results = []
    for strategy in ["most_frequent", "uniform"]:
        result = train_dummy_strategy(
            X_train, X_test, y_train, y_test, strategy, config
        )
        results.append(result)

    # As métricas devem ser diferentes para estratégias diferentes
    assert (
        results[0]["metrics"]["accuracy"] != results[1]["metrics"]["accuracy"]
    )


def test_run_all_strategies_returns_dataframe(monkeypatch: object) -> None:
    """run_all_strategies deve retornar DataFrame com 3 estratégias."""
    df = make_dummy_df()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test = X.iloc[:3], X.iloc[3:]
    y_train, y_test = y.iloc[:3], y.iloc[3:]

    config = DummyTrainingConfig(random_seed=42)

    # Mock do MLflow
    class _DummyRun:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.start_run",
        lambda run_name=None: _DummyRun(),
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.set_tag",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.sklearn.log_model",
        lambda *args, **kwargs: None,
    )

    results_df = run_all_strategies(
        X_train, X_test, y_train, y_test, config, dataset_version="test123"
    )

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == EXPECTED_STRATEGIES_COUNT
    assert set(results_df["strategy"].tolist()) == {
        "most_frequent",
        "stratified",
        "uniform",
    }
    # DataFrame deve estar ordenado por f1_score
    assert results_df["f1_score"].is_monotonic_decreasing


def test_run_all_strategies_logs_to_mlflow(monkeypatch: object) -> None:
    """run_all_strategies deve fazer log no MLflow para cada estratégia."""
    df = make_dummy_df()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test = X.iloc[:3], X.iloc[3:]
    y_train, y_test = y.iloc[:3], y.iloc[3:]

    config = DummyTrainingConfig(random_seed=42)

    logged_params: list[dict[str, object]] = []
    logged_metrics: list[dict[str, object]] = []

    class _DummyRun:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def mock_log_param(key: str, value: object) -> None:
        if not logged_params or key in logged_params[-1]:
            logged_params.append({})
        logged_params[-1][key] = value

    def mock_log_metric(key: str, value: object) -> None:
        if not logged_metrics or key in logged_metrics[-1]:
            logged_metrics.append({})
        logged_metrics[-1][key] = value

    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.start_run",
        lambda run_name=None: _DummyRun(),
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_param",
        mock_log_param,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.set_tag",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.log_metric",
        mock_log_metric,
    )
    monkeypatch.setattr(
        "src.training.dummy_trainer.mlflow.sklearn.log_model",
        lambda *args, **kwargs: None,
    )

    run_all_strategies(X_train, X_test, y_train, y_test, config)

    # Deve ter feito log para 3 estratégias
    assert len(logged_params) >= EXPECTED_STRATEGIES_COUNT
    assert len(logged_metrics) >= EXPECTED_STRATEGIES_COUNT

"""Testes para o pipeline run_logistic_regression."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import pytest

import mlflow.sklearn as _mlflow_sklearn
from src.pipelines.run_logistic_regression import main

_N = 20
_N_FEATURES = 4
_RNG = np.random.default_rng(42)
_X = _RNG.random((_N, _N_FEATURES)).astype(np.float32)
_Y = np.array([0, 1] * (_N // 2), dtype=np.float32)
_FEATURE_NAMES = [f"feat_{i}" for i in range(_N_FEATURES)]

_CV_RESULTS = {
    "cv_accuracy_mean": 0.70,
    "cv_accuracy_std": 0.05,
    "cv_precision_mean": 0.65,
    "cv_precision_std": 0.05,
    "cv_recall_mean": 0.60,
    "cv_recall_std": 0.05,
    "cv_f1_mean": 0.62,
    "cv_f1_std": 0.05,
    "cv_roc_auc_mean": 0.72,
    "cv_roc_auc_std": 0.05,
}

_TEST_METRICS = {
    "accuracy": 0.75,
    "precision": 0.70,
    "recall": 0.65,
    "f1_score": 0.67,
    "roc_auc": 0.78,
    "pr_auc": 0.60,
}


class _FakeModel:
    coef_ = np.zeros((1, _N_FEATURES))


class _DummyRun:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: object, exc: object, tb: object
    ) -> None:
        return None


def test_main_returns_zero_with_monkeypatched_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main deve retornar 0 quando dependencias externas sao mockadas."""
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.load_telco_data",
        lambda _: pd.DataFrame(
            {"col": [1, 2], "Churn": ["Yes", "No"]}
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.setup_mlflow",
        lambda _: None,
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.mlp_preprocess_data",
        lambda _: (_X, _Y, _FEATURE_NAMES, pd.DataFrame(_X)),
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.cross_validate_logistic",
        lambda *args, **kwargs: _CV_RESULTS,
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.train_logistic_classifier",
        lambda *args, **kwargs: {
            "model": _FakeModel(),
            "metrics": _TEST_METRICS,
        },
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.mlflow.start_run",
        lambda: _DummyRun(),
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.mlflow.log_params",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.mlflow.set_tag",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.mlflow.log_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        _mlflow_sklearn,
        "log_model",
        lambda *args, **kwargs: None,
    )

    assert main([]) == 0


def test_main_raises_on_missing_target_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main deve propagar ValueError quando coluna Churn estiver ausente."""
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.load_telco_data",
        lambda _: pd.DataFrame({"col": [1, 2]}),
    )
    monkeypatch.setattr(
        "src.pipelines.run_logistic_regression.setup_mlflow",
        lambda _: None,
    )

    with pytest.raises(ValueError, match="Coluna alvo ausente"):
        main([])

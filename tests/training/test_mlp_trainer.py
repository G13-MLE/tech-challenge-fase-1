from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.models import MLPConfig, TrainingConfig
from src.training import MLPForTraining, MLPTrainer
from src.training.mlp.trainer import cross_validate_mlp

EXPECTED_CV_KEYS = {
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_precision_mean",
    "cv_precision_std",
    "cv_recall_mean",
    "cv_recall_std",
    "cv_f1_score_mean",
    "cv_f1_score_std",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "cv_pr_auc_mean",
    "cv_pr_auc_std",
    "cv_brier_score_mean",
    "cv_brier_score_std",
}


@pytest.mark.slow
def test_overfit_single_batch(tmp_path: Path) -> None:
    config = MLPConfig(input_dim=10, hidden_dims=(64, 32))
    model = MLPForTraining(config)

    training_config = TrainingConfig(
        optimizer="adam",
        lr=0.01,
        weight_decay=0.0,
        scheduler=None,
        batch_size=32,
        max_epochs=20,
        early_stopping_patience=10,
        val_split=0.2,
    )

    trainer = MLPTrainer(model, training_config, device="cpu")

    np.random.seed(42)
    X = np.random.randn(32, 10).astype(np.float32)
    y = np.random.randint(0, 2, 32).astype(np.float32)

    model_path = tmp_path / "test_model.pt"
    history = trainer.fit(X, y, model_save_path=str(model_path))

    assert history["train_loss"][-1] < history["train_loss"][0]


@pytest.mark.slow
def test_fit_returns_history() -> None:
    config = MLPConfig(input_dim=10, hidden_dims=(32,))
    model = MLPForTraining(config)

    training_config = TrainingConfig(
        optimizer="adam",
        lr=0.01,
        batch_size=16,
        max_epochs=5,
        early_stopping_patience=10,
    )

    trainer = MLPTrainer(model, training_config, device="cpu")

    X = np.random.randn(64, 10).astype(np.float32)
    y = np.random.randint(0, 2, 64).astype(np.float32)

    history = trainer.fit(X, y)

    assert "train_loss" in history
    assert "val_loss" in history
    assert "val_f1" in history
    assert "val_auc" in history
    assert len(history["train_loss"]) > 0


@pytest.mark.slow
def test_cross_validate_mlp_returns_expected_keys() -> None:
    """cross_validate_mlp deve retornar dict com todas as chaves
    de metricas CV agregadas (media e std)."""
    np.random.seed(42)
    X = np.random.randn(80, 10).astype(np.float32)
    y = np.array([0, 1] * 40, dtype=np.float32)

    mlp_config = MLPConfig(input_dim=10, hidden_dims=(16,))
    training_config = TrainingConfig(
        optimizer="adam",
        lr=0.01,
        batch_size=16,
        max_epochs=3,
        early_stopping_patience=10,
        val_split=0.2,
        random_seed=42,
    )

    cv_results = cross_validate_mlp(
        X, y, mlp_config, training_config, n_folds=2
    )

    assert set(cv_results.keys()) == EXPECTED_CV_KEYS


@pytest.mark.slow
def test_cross_validate_mlp_values_in_range() -> None:
    """Metricas do CV MLP devem estar entre 0 e 1."""
    np.random.seed(42)
    X = np.random.randn(80, 10).astype(np.float32)
    y = np.array([0, 1] * 40, dtype=np.float32)

    mlp_config = MLPConfig(input_dim=10, hidden_dims=(16,))
    training_config = TrainingConfig(
        optimizer="adam",
        lr=0.01,
        batch_size=16,
        max_epochs=3,
        early_stopping_patience=10,
        val_split=0.2,
        random_seed=42,
    )

    cv_results = cross_validate_mlp(
        X, y, mlp_config, training_config, n_folds=2
    )

    for k, v in cv_results.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} fora do intervalo [0, 1]"

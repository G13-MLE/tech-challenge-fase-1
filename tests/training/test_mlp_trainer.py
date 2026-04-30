from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.models import MLPConfig, TrainingConfig
from src.training import MLPForTraining, MLPTrainer


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

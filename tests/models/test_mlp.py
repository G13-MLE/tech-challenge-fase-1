from __future__ import annotations

import pytest
import torch

from src.config.models import MLPConfig
from src.training import MLP, MLPForTraining


@pytest.mark.fast
def test_forward_output_shape() -> None:
    config = MLPConfig(input_dim=10, hidden_dims=(64, 32))
    model = MLP(config)
    x = torch.randn(5, 10)
    output = model(x)

    assert output.shape == (5, 1)
    assert (output >= 0).all()
    assert (output <= 1).all()


@pytest.mark.fast
def test_forward_different_batch_sizes() -> None:
    config = MLPConfig(
        input_dim=10, hidden_dims=(64, 32), use_batch_norm=False
    )
    model = MLP(config)

    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, 10)
        output = model(x)
        assert output.shape == (batch_size, 1)


@pytest.mark.fast
def test_parameter_count() -> None:
    config = MLPConfig(input_dim=10, hidden_dims=(8, 4), use_batch_norm=False)
    model = MLP(config)

    expected = 10 * 8 + 8 + 8 * 4 + 4 + 4 * 1 + 1

    assert model.get_num_parameters() == expected


@pytest.mark.fast
def test_config_frozen() -> None:
    config = MLPConfig(input_dim=10)

    with pytest.raises(AttributeError):
        config.input_dim = 20


@pytest.mark.fast
def test_mlp_for_training_returns_loss() -> None:
    config = MLPConfig(input_dim=10, hidden_dims=(32,))
    model = MLPForTraining(config)
    x = torch.randn(5, 10)
    targets = torch.randint(0, 2, (5,), dtype=torch.float32)

    outputs = model(x, targets=targets)

    assert "loss" in outputs
    assert "logits" in outputs
    assert "probs" in outputs
    assert outputs["loss"].item() > 0
    assert (outputs["probs"] >= 0).all()
    assert (outputs["probs"] <= 1).all()

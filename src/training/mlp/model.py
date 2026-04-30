"""Arquitetura MLP para classificacao binaria."""

from __future__ import annotations

import torch
from torch import nn

from src.config.models import MLPConfig


class MLP(nn.Module):
    """Perceptron Multicamadas para inferencia.

    Aplica sigmoid na saida, produzindo probabilidades em [0, 1].
    Usar MLPForTraining para treino.

    Attributes:
        config: MLPConfig com parametros da arquitetura.
        hidden_layers: Camadas ocultas (Linear, BN, ReLU, Dropout).
        output_layer: Camada Linear de saida.
        sigmoid: Ativacao Sigmoid.
    """

    def __init__(self, config: MLPConfig) -> None:
        """Inicializa a arquitetura MLP.

        Args:
            config: Configuracao da arquitetura.
        """
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass retornando probabilidades.

        Args:
            x: Tensor de entrada (batch_size, input_dim).

        Returns:
            Probabilidades (batch_size,).
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        return self.sigmoid(logits)

    def get_num_parameters(self) -> int:
        """Conta parametros treinaveis."""
        return sum(p.numel() for p in self.parameters())


class MLPForTraining(nn.Module):
    """Wrapper de treino com BCEWithLogitsLoss.

    Usa BCEWithLogitsLoss para estabilidade numerica.

    Attributes:
        config: MLPConfig.
        model: Instancia MLP subjacente.
        criterion: Funcao de perda.
    """

    def __init__(self, config: MLPConfig) -> None:
        """Inicializa o wrapper de treino.

        Args:
            config: Configuracao da arquitetura.
        """
        super().__init__()
        self.config = config
        self.model = MLP(config)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass retornando logits, probs e perda opcional.

        Args:
            x: Tensor de entrada.
            targets: Rotulos opcionais.

        Returns:
            Dicionario com logits, probs e loss (se targets fornecido).
        """
        hidden = self.model.hidden_layers(x)
        logits = self.model.output_layer(hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": probs,
        }

        if targets is not None:
            loss = self.criterion(logits, targets.float())
            result["loss"] = loss

        return result

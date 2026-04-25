"""Dataclasses de configuracao para MLP e hiperparametros de treino."""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import RANDOM_SEED


@dataclass(frozen=True)
class MLPConfig:
    """Configuracao da arquitetura MLP.

    Attributes:
        input_dim: Numero de features de entrada.
        hidden_dims: Tupla com tamanho de cada camada oculta.
        dropout_rate: Probabilidade de dropout durante treino.
        use_batch_norm: Se usa BatchNorm apos cada Linear.
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.3
    use_batch_norm: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparametros para treinamento.

    Attributes:
        optimizer: Algoritmo de otimizacao ("adam" ou "sgd").
        lr: Learning rate.
        weight_decay: Coeficiente de regularizacao L2.
        scheduler: Estrategia de LR scheduler ou None.
        scheduler_patience: Epocas antes de reduzir LR
            (para "reduce_on_plateau").
        early_stopping_patience: Epocas sem melhoria antes de parar.
        early_stopping_min_delta: Mudanca minima para considerar melhoria.
        batch_size: Amostras por batch.
        max_epochs: Limite maximo de epocas.
        val_split: Fracao dos dados de treino para validacao.
        random_seed: Semente para reprodutibilidade.
    """

    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str | None = "reduce_on_plateau"
    scheduler_patience: int = 3
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    val_split: float = 0.2
    random_seed: int = RANDOM_SEED

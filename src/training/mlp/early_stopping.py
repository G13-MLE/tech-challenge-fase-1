"""Mecanismo de parada antecipada (early stopping)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Callback de parada antecipada.

    Para o treino quando a metrica de validacao para de melhorar
    por um numero configuravel de epocas (patience).

    Attributes:
        patience: Epocas de espera por melhoria antes de parar.
        min_delta: Mudanca minima para qualificar como melhoria.
        mode: "min" para perda, "max" para metricas.
        counter: Contagem atual de epocas sem melhoria (interno).
        best_score: Melhor pontuacao ate agora (interno).
        early_stop: Flag indicando se o treino deve parar (interno).
    """

    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    counter: int = field(default=0, init=False)
    best_score: float | None = field(default=None, init=False)
    early_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Valida configuracao de mode."""
        if self.mode not in {"min", "max"}:
            raise ValueError(
                f"mode deve ser 'min' ou 'max', recebeu {self.mode}"
            )

    def __call__(self, score: float) -> bool:
        """Verifica se o treino deve parar.

        Args:
            score: Valor atual da metrica de validacao.

        Returns:
            True se o treino deve parar.
        """
        if self.is_best(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def is_best(self, score: float) -> bool:
        """Verifica se a pontuacao representa uma nova melhoria.

        Args:
            score: Valor atual da metrica.

        Returns:
            True se for melhor que o anterior.
        """
        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def reset(self) -> None:
        """Reseta o estado interno para nova execucao."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

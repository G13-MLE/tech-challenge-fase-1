"""Monitoramento de data drift por janela usando PSI real.

Acumula amostras em buffer circular e periodicamente calcula
PSI (Population Stability Index) comparando a distribuicao da
janela contra a baseline de treino.

Diferente da deteccao per-request (que flagra anomalias pontuais
como out-of-range), este modulo mede deslocamentos distribucionais
com significancia estatistica.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Thresholds padrao de PSI
_PSI_STABLE = 0.1
_PSI_MODERATE = 0.25

# Tamanho padrao da janela circular
_DEFAULT_WINDOW_SIZE = 200

# Epsilon para evitar log(0)
_EPSILON = 1e-6


@dataclass
class PsiWindow:
    """Buffer circular para acumulo de amostras e calculo de PSI.

    Mantem as ultimas N amostras de uma feature numerica e compara
    a distribuicao da janela contra bins de referencia da baseline
    de treino.
    """

    feature: str
    buffer: deque[float] = field(default_factory=deque)
    max_size: int = _DEFAULT_WINDOW_SIZE

    @classmethod
    def new(
        cls,
        feature: str,
        max_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> PsiWindow:
        """Factory para criar janela com tamanho especifico."""
        return cls(
            feature=feature,
            buffer=deque(maxlen=max_size),
            max_size=max_size,
        )

    def add(self, value: float) -> None:
        """Adiciona um valor numerico ao buffer."""
        self.buffer.append(value)

    @property
    def size(self) -> int:
        """Quantidade atual de amostras no buffer."""
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        """True se o buffer atingiu o tamanho minimo para PSI."""
        return len(self.buffer) >= self.max_size

    def compute_psi(self, baseline_bins: list[dict[str, Any]]) -> float:
        """Calcula PSI comparando distribuicao da janela vs baseline.

        Formula:
            PSI = sum((actual_i - expected_i) * ln(actual_i / expected_i))

        Args:
            baseline_bins: Lista de bins de referencia com chaves
                "lower", "upper" e "proportion".

        Returns:
            PSI score arredondado com 4 casas decimais.
        """
        total = len(self.buffer)
        if total == 0:
            return 0.0

        psi = 0.0

        for b in baseline_bins:
            lower = b["lower"]
            upper = b["upper"]
            expected = b["proportion"]

            # Proporcao observada na janela para este bin
            actual = sum(1 for v in self.buffer if lower <= v <= upper) / total

            # Evita divisao por zero / log(0)
            actual = max(actual, _EPSILON)
            expected = max(expected, _EPSILON)

            psi += (actual - expected) * math.log(actual / expected)

        return round(psi, 4)

    def reset(self) -> None:
        """Limpa o buffer para iniciar nova janela."""
        self.buffer.clear()


@dataclass
class PsiResult:
    """Resultado do calculo de PSI para uma feature."""

    feature: str
    score: float
    status: str  # "stable", "moderate", "significant"

    @classmethod
    def from_window(
        cls,
        window: PsiWindow,
        baseline_bins: list[dict[str, Any]],
    ) -> PsiResult:
        """Cria PsiResult a partir de uma PsiWindow e baseline."""
        score = window.compute_psi(baseline_bins)

        if score < _PSI_STABLE:
            status = "stable"
        elif score < _PSI_MODERATE:
            status = "moderate"
        else:
            status = "significant"

        return cls(feature=window.feature, score=score, status=status)

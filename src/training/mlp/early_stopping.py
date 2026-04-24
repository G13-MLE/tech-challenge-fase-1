"""Mecanismo de parada antecipada (early stopping) para prevenir
overfitting durante o treino.

Early stopping monitora uma métrica de validação e para o treino
quando nenhuma melhoria é observada por um número configurável
de épocas (paciência). Isso previne overfitting parando antes
que o modelo comece a memorizar os dados de treino.

Por que Early Stopping Funciona:
- A perda de treino tipicamente continua diminuindo enquanto
  a perda de validação eventualmente aumenta
- A diferença entre perda de treino e validação indica overfitting
- Parar no momento certo produz melhor generalização

A classe EarlyStopping mantém estado - ela rastreia:
- Melhor pontuação de validação alcançada até agora
- Número de épocas sem melhoria (contador)
- Se o treino deve parar

Este é um padrão padrão em deep learning para melhorar a
generalização do modelo e reduzir o tempo de treino.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Callback de parada antecipada que monitora métricas de validação.

    Para o treino quando a métrica de validação para de melhorar por
    uma paciência especificada. Esta é uma técnica comum de
    regularização que previne overfitting abortando o treino antes
    que o modelo comece a memorizar os dados de treino.

    A classe mantém estado e rastreia a melhor pontuação e contador
    internamente. Chame-a após cada época de validação para verificar
    se o treino deve parar.

    Attributes (Atributos):
        patience: Número de épocas para esperar por melhoria antes
            de parar. Valores típicos: 3-10. Valores maiores permitem
            mais "platô" antes de parar.
        min_delta: Mudança mínima na pontuação para qualificar como
            melhoria. Valores maiores exigem melhorias mais
            significativas, prevenindo resets por ruído.
            Típico: 0.0001 a 0.01.
        mode: "min" para perda (menor é melhor), "max" para
            métricas como acurácia/F1. Determina como comparar
            pontuações para melhoria.
        counter: Contagem atual de épocas sem melhoria (interno).
        best_score: Melhor pontuação alcançada até agora (interno).
        early_stop: Flag indicando se o treino deve parar (interno).

    Exemplo:
        >>> early_stop = EarlyStopping(patience=5, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     if early_stop(val_loss):
        ...         print(f"Parando na época {epoch}")
        ...         break

    Note:
        - Use mode="min" para perdas (validation loss)
        - Use mode="max" para métricas (accuracy, F1-score, AUC)
        - Resetar o estado ao reutilizar a mesma instância
    """

    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    counter: int = field(default=0, init=False)
    best_score: float | None = field(default=None, init=False)
    early_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Valida configuração após inicialização.

        Garante que o modo seja um dos valores suportados para prevenir erros
        silenciosos durante a comparação de pontuações.
        """
        if self.mode not in {"min", "max"}:
            raise ValueError(
                f"mode deve ser 'min' ou 'max', recebeu {self.mode}"
            )

    def __call__(self, score: float) -> bool:
        """Verifica se o treino deve parar após observar a pontuação.

        Esta é a interface principal - chame após cada época de
        validação. Atualiza o estado interno e retorna se a
        parada antecipada foi acionada.

        Args:
            score: Valor atual da métrica de validação (perda
                ou métrica).

        Returns:
            True se o treino deve parar (sem melhoria por
            `patience` épocas). False se o treino deve continuar.

        Mudanças de Estado:
            - Se nova melhor pontuação: reseta contador para 0,
              atualiza best_score
            - Se não é melhor: incrementa contador, pode definir
              early_stop como True
        """
        if self.is_best(score):
            # Melhoria encontrada: reseta contador e atualiza melhor pontuação
            self.best_score = score
            self.counter = 0
        else:
            # Sem melhoria: incrementa contador em direção à parada antecipada
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def is_best(self, score: float) -> bool:
        """Verifica se a pontuação representa um novo valor máximo.

        Compara a pontuação atual contra best_score, considerando
        a direção (min vs max) e limiar de melhoria mínima.

        Args:
            score: Valor atual da métrica de validação.

        Returns:
            True se esta for a primeira pontuação ou mostrar
            melhoria sobre o anterior.
        """
        # Primeira pontuação é sempre a "melhor" por padrão
        if self.best_score is None:
            return True

        # Modo determina direção de comparação:
        # - "min": Pontuação deve ser menor que o anterior (perdas)
        # - "max": Pontuação deve ser maior que o anterior (métricas)
        if self.mode == "min":
            # Para perda: melhoria significa pontuação caiu min_delta
            return score < self.best_score - self.min_delta
        else:
            # Para métricas: melhoria significa pontuação subiu min_delta
            return score > self.best_score + self.min_delta

    def reset(self) -> None:
        """Reseta todo o estado interno para uma nova execução de treino.

        Chame isso ao reutilizar a mesma instância EarlyStopping para múltiplas
        execuções de treino ou ao realizar validação cruzada.
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False

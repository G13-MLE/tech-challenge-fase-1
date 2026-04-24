"""Dataclasses de configuração para MLP e hiperparâmetros de treino.

Este módulo fornece dataclasses imutáveis que encapsulam todos os
parâmetros de configuração necessários para criação e treino do
modelo. Usar dataclasses frozen garante imutabilidade e previne
modificação acidental durante o treino.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import RANDOM_SEED


@dataclass(frozen=True)
class MLPConfig:
    """Configuração da arquitetura MLP (Perceptron Multicamadas).

    Esta configuração define a estrutura da rede neural. A arquitetura
    segue o padrão: Linear -> (BatchNorm) -> ReLU -> Dropout para
    cada camada oculta, terminando com uma camada Linear de saída.

    Attributes (Atributos):
        input_dim: Número de features de entrada (deve corresponder
            ao formato dos dados pré-processados).
        hidden_dims: Tupla definindo o tamanho de cada camada oculta.
            Mais camadas e dimensões maiores aumentam a capacidade do
            modelo, mas podem levar ao overfitting.
        dropout_rate: Probabilidade de zerar ativações durante o
            treino. Valores maiores (0.3-0.5) fornecem maior
            regularização contra overfitting. Use 0.0 para inferência.
        use_batch_norm: Se deve usar BatchNorm após cada camada Linear.
            BatchNorm estabiliza o treino normalizando ativações entre
            camadas, permitindo learning rates maiores e convergência
            mais rápida. Também fornece um efeito leve de regularização.

    Exemplo:
        >>> config = MLPConfig(
        ...     input_dim=45,  # Após pré-processamento
        ...     hidden_dims=(128, 64, 32),  # 3 camadas
        ...     dropout_rate=0.3,
        ...     use_batch_norm=True,
        ... )
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.3
    use_batch_norm: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros para o processo de treinamento.

    Esta configuração controla a otimização, agendamento do learning
    rate, parada antecipada (early stopping) e comportamento de
    carregamento dos dados. Esses parâmetros afetam diretamente a
    velocidade de convergência e o desempenho final do modelo.

    Attributes (Atributos):
        optimizer: Algoritmo de otimização. "adam" é recomendado para
            a maioria dos casos devido à sua taxa de aprendizado
            adaptativa por parâmetro. "sgd" pode funcionar melhor
            para alguns problemas, mas requer mais ajustes.
        lr: Learning rate que controla o tamanho do passo do
            gradiente. Muito alto causa divergência, muito baixo
            leva a convergência lenta. Intervalo típico: 1e-4 a 1e-2.
        weight_decay: Coeficiente de regularização L2. Previne
            overfitting penalizando pesos grandes. Valores maiores
            (1e-3) para modelos complexos, menores (1e-5) para mais
            simples.
        scheduler: Estratégia de agendamento do learning rate.
            "reduce_on_plateau" diminui o LR quando a perda de
            validação para de melhorar. "step" diminui em
            intervalos fixos. None para LR fixo.
        scheduler_patience: Épocas de espera antes de reduzir o
            LR (para "reduce_on_plateau"). Valores maiores permitem
            mais épocas antes da intervenção.
        early_stopping_patience: Épocas de espera por melhoria
            antes de parar. Previne overfitting parando quando a
            perda de validação para de diminuir.
        early_stopping_min_delta: Mudança mínima para qualificar
            como melhoria. Previne parada por flutuações de ruído.
            Valores maiores exigem melhorias mais significativas.
        batch_size: Amostras por atualização do gradiente. Batches
            maiores fornecem gradientes mais estáveis mas exigem
            mais memória. Batches menores adicionam ruído de
            regularização, mas podem ser instáveis.
        max_epochs: Máximo de épocas de treino. Atua como limite
            de segurança quando early stopping está desabilitado.
        val_split: Fração dos dados de treino para validação
            (0.0-1.0). Usada para early stopping e seleção do
            modelo quando X_val não é fornecido.
        random_seed: Semente para reprodutibilidade. Garante que
            divisões treino/val e inicialização sejam consistentes.

    Exemplo:
        >>> config = TrainingConfig(
        ...     optimizer="adam",
        ...     lr=0.001,
        ...     early_stopping_patience=5,  # 5 épocas
        ... )
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

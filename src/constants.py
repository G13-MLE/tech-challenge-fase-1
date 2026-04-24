"""Constantes compartilhadas entre pipelines de ML.

Este modulo centraliza valores constantes usados em multiplos
modulos para garantir consistencia e facilitar manutencao.
"""

from __future__ import annotations

# Seed para reprodutibilidade
RANDOM_SEED: int = 42

# Coluna alvo para predição de churn
TARGET_COLUMN: str = "Churn"

# Label positivo para churn (Yes = cliente cancelou)
POSITIVE_LABEL: str = "Yes"

# Limiar para converter probabilidades em predicoes binarias
THRESHOLD: float = 0.5

# Thresholds para bandas de risco do Canvas de ML
RISK_BAND_LOW: float = 0.30
RISK_BAND_HIGH: float = 0.60

# Proporcao padrao para split treino/teste
# Proporcao padrão para split treino/teste
DEFAULT_TEST_SIZE: float = 0.2

# Caminho padrão para o dataset
DEFAULT_DATASET_PATH: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Caminho padrão para salvar modelos
DEFAULT_MODEL_DIR: str = "models"

# Nome padrão para experimento MLflow
DEFAULT_MLP_EXPERIMENT_NAME: str = "tech-challenge-mlp"
DEFAULT_DUMMY_EXPERIMENT_NAME: str = "tech-challenge-dummy-baseline"

# Latencia SLO threshold em milissegundos
LATENCY_SLO_THRESHOLD_MS: float = 500.0

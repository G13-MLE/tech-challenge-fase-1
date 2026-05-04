"""Modulo de inferencia para o modelo MLP de churn."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src.config.models import MLPConfig
from src.data.cleaning import clean_telco_data
from src.data.preprocessing import apply_scaling, load_scaler
from src.training.mlp.model import MLP

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/churn_mlp_best.pt")
DEFAULT_SCALER_PATH = Path("models/scaler.pkl")
DEFAULT_FEATURE_NAMES_PATH = Path("models/feature_names.json")

# Categorias na ordem de aparecimento do dataset de treino.
# Usadas para garantir colunas dummy consistentes na inferencia.
_CATEGORICAL_COLUMNS: dict[str, list[str]] = {
    "gender": sorted(["Female", "Male"]),
    "Partner": sorted(["Yes", "No"]),
    "Dependents": sorted(["No", "Yes"]),
    "PhoneService": sorted(["No", "Yes"]),
    "MultipleLines": sorted(["No phone service", "No", "Yes"]),
    "InternetService": sorted(["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": sorted(["No", "Yes", "No internet service"]),
    "OnlineBackup": sorted(["Yes", "No", "No internet service"]),
    "DeviceProtection": sorted(["No", "Yes", "No internet service"]),
    "TechSupport": sorted(["No", "Yes", "No internet service"]),
    "StreamingTV": sorted(["No", "Yes", "No internet service"]),
    "StreamingMovies": sorted(["No", "Yes", "No internet service"]),
    "Contract": sorted(["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": sorted(["Yes", "No"]),
    "PaymentMethod": sorted(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    ),
}


def _infer_mlp_config(
    state_dict: dict[str, torch.Tensor],
) -> MLPConfig:
    """Infere a configuracao da arquitetura MLP a partir do state_dict.

    Args:
        state_dict: Estado do modelo salvo por save_best_model.

    Returns:
        MLPConfig reconstruido com input_dim, hidden_dims e
        use_batch_norm.
    """
    first_weight = state_dict["hidden_layers.0.weight"]
    input_dim = int(first_weight.shape[1])

    linear_weights: list[tuple[int, int]] = []
    pattern = re.compile(r"hidden_layers\.(\d+)\.weight$")

    _tensor_dim_linear = 2
    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if match and tensor.ndim == _tensor_dim_linear:
            linear_weights.append((int(match.group(1)), int(tensor.shape[0])))

    linear_weights.sort(key=lambda item: item[0])
    hidden_dims = tuple(shape for _, shape in linear_weights)

    use_batch_norm = any(
        tensor.ndim == 1
        for key, tensor in state_dict.items()
        if pattern.match(key)
    )

    return MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=0.0,
        use_batch_norm=use_batch_norm,
    )


def _load_feature_names(path: Path) -> list[str]:
    """Carrega a lista ordenada de features do JSON.

    Args:
        path: Caminho para o arquivo feature_names.json.

    Returns:
        Lista com nomes das features na ordem do treino.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_dataframe(customer_data: dict[str, Any]) -> pd.DataFrame:
    """Monta DataFrame de uma unica linha a partir dos dados do cliente.

    Preenche TotalCharges quando ausente ou invalido para evitar
    remocao da linha durante a limpeza.

    Args:
        customer_data: Dicionario com os campos brutos do cliente.

    Returns:
        DataFrame limpo pronto para preprocessamento.
    """
    customer_data = dict(customer_data)
    if "Churn" not in customer_data:
        customer_data["Churn"] = "No"

    df = pd.DataFrame([customer_data])

    if "TotalCharges" not in df.columns or df["TotalCharges"].isna().all():
        _monthly_series = df.get("MonthlyCharges", pd.Series([0.0]))
        _tenure_series = df.get("tenure", pd.Series([0]))
        monthly = float(_monthly_series.iloc[0])
        tenure_val = int(_tenure_series.iloc[0])
        df["TotalCharges"] = monthly * tenure_val if tenure_val > 0 else 0.0
    else:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    return clean_telco_data(df)


def _preprocess_for_inference(
    df: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Preprocessa DataFrame para inferencia garantindo colunas fixas.

    Aplica one-hot encoding com categorias pre-definidas para evitar
    divergencia entre treino e inferencia em DataFrames de 1 linha.

    Args:
        df: DataFrame limpo com uma unica linha.
        feature_names: Lista de features esperadas pelo modelo.

    Returns:
        Array numpy de shape (1, n_features) pronto para scaling.
    """
    df_work = df.copy()

    # Converte categoricas para Categorical com ordem fixa
    for col, categories in _CATEGORICAL_COLUMNS.items():
        if col in df_work.columns:
            df_work[col] = pd.Categorical(df_work[col], categories=categories)

    # Aplica one-hot encoding apenas nas colunas presentes
    present_categorical = [
        col for col in _CATEGORICAL_COLUMNS if col in df_work.columns
    ]
    df_encoded = pd.get_dummies(
        df_work,
        columns=present_categorical,
        drop_first=True,
    )

    # Garante que todas as features esperadas existam
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reordena para a mesma ordem do treino
    df_encoded = df_encoded[feature_names]

    return np.asarray(df_encoded.values, dtype=np.float64)


class ChurnPredictor:
    """Wrapper para carregamento lazy e predicao do modelo MLP."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
    ) -> None:
        self._model_path = Path(model_path)
        self._scaler_path = Path(scaler_path)
        self._feature_names_path = Path(feature_names_path)
        self._model: torch.nn.Module | None = None
        self._scaler: BaseEstimator | None = None
        self._feature_names: list[str] = []
        self._device = torch.device("cpu")

    def load(self) -> None:
        """Carrega modelo, scaler e feature names do disco."""
        if self._model is not None:
            return

        for path, label in [
            (self._model_path, "Modelo"),
            (self._scaler_path, "Scaler"),
            (self._feature_names_path, "Feature names"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} nao encontrado: {path}")

        checkpoint = torch.load(
            self._model_path,
            weights_only=False,
            map_location=self._device,
        )
        state_dict = checkpoint["model_state_dict"]

        config = _infer_mlp_config(state_dict)
        model = MLP(config)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()

        self._model = model
        self._scaler = load_scaler(str(self._scaler_path))
        self._feature_names = _load_feature_names(self._feature_names_path)

        logger.info(
            "Modelo carregado: input_dim=%d hidden_dims=%s",
            config.input_dim,
            config.hidden_dims,
        )

    def predict(self, customer_data: dict[str, Any]) -> float:
        """Retorna a probabilidade de churn para um unico cliente.

        Args:
            customer_data: Dicionario com todos os campos necessarios
                do dataset Telco (exceto customerID e Churn).

        Returns:
            Probabilidade de churn no intervalo [0.0, 1.0].

        Raises:
            RuntimeError: Se o modelo nao foi carregado.
            ValueError: Se os dados falharem na validacao.
        """
        self.load()

        if self._model is None or self._scaler is None:
            raise RuntimeError("Modelo nao carregado corretamente")

        df_clean = _prepare_dataframe(customer_data)

        if len(df_clean) == 0:
            raise ValueError("Dados invalidos: falha na validacao de dominio")

        X = _preprocess_for_inference(df_clean, self._feature_names)
        X_scaled = apply_scaling(X, self._scaler)

        tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            probability = self._model(tensor).squeeze().item()

        return float(probability)


class _PredictorHolder:
    """Holder para singleton de ChurnPredictor."""

    instance: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Retorna singleton do ChurnPredictor."""
    if _PredictorHolder.instance is None:
        _PredictorHolder.instance = ChurnPredictor()
    return _PredictorHolder.instance


def predict_single(customer_data: dict[str, Any]) -> float:
    """Interface de conveniencia para predicao unitaria."""
    return get_predictor().predict(customer_data)

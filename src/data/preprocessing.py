"""Preprocessamento de dados para modelos MLP.

Este modulo fornece funcoes de preprocessamento especificas
para preparacao de dados tabulares para redes neurais.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def remove_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a coluna customerID do DataFrame se existir.

    Args:
        df: DataFrame com ou sem coluna customerID

    Returns:
        DataFrame sem a coluna customerID (se existia)
    """
    if "customerID" in df.columns:
        return df.drop(columns=["customerID"])
    return df.copy()


def encode_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Codifica coluna target para valores binarios.

    Args:
        df: DataFrame com coluna target
        target_col: Nome da coluna target (default: "Churn")

    Returns:
        DataFrame com target codificado (Yes->1, No->0)
    """
    df_result = df.copy()
    df_result[target_col] = df_result[target_col].map({"Yes": 1, "No": 0})
    return df_result


def clean_total_charges(
    df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Converte TotalCharges para numerico e preenche NaNs.

    TotalCharges as vezes vem como string vazia no dataset Telco.

    Args:
        df: DataFrame com coluna TotalCharges
        fill_value: Valor para preencher NaNs (default: 0.0)

    Returns:
        DataFrame com TotalCharges como float e NaNs preenchidos
    """
    df_result = df.copy()
    df_result["TotalCharges"] = pd.to_numeric(
        df_result["TotalCharges"],
        errors="coerce",
    )
    df_result["TotalCharges"] = df_result["TotalCharges"].fillna(fill_value)
    return df_result


def one_hot_encode(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Aplica one-hot encoding em colunas categoricas.

    Args:
        df: DataFrame com colunas categoricas
        categorical_cols: Lista de colunas a codificar. Se None,
            detecta automaticamente colunas do tipo object
        drop_first: Se True, elimina primeira categoria para
            evitar multicolinearidade

    Returns:
        DataFrame com colunas categoricas codificadas
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not categorical_cols:
        return df.copy()

    return pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=drop_first,
    )


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separa DataFrame em features (X) e target (y).

    Args:
        df: DataFrame com target já codificado
        target_col: Nome da coluna target (default: "Churn")

    Returns:
        Tupla contendo:
            - X: DataFrame com features
            - y: Series com target
            - feature_names: Lista com nomes das features
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, X.columns.tolist()


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Cria e fita um StandardScaler nos dados de treino.

    IMPORTANTE: Sempre fit apenas no conjunto de treino para
    evitar data leakage. Use transform() nos dados de teste.

    Args:
        X_train: Features de treino de shape (n_samples, n_features)

    Returns:
        StandardScaler fitado nos dados de treino
    """
    scaler: BaseEstimator = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaling(
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Aplica scaling em features usando scaler pre-fitado.

    Args:
        X: Array de features de shape (n_samples, n_features)
        scaler: StandardScaler já fitado nos dados de treino

    Returns:
        Array de features escaladas
    """
    return scaler.transform(X)


def save_scaler(scaler: StandardScaler, filepath: str) -> None:
    """Salva um scaler em arquivo.

    Args:
        scaler: StandardScaler fitado
        filepath: Caminho para salvar o arquivo
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)


def load_scaler(filepath: str) -> StandardScaler:
    """Carrega um scaler de arquivo.

    Args:
        filepath: Caminho para o arquivo do scaler

    Returns:
        StandardScaler carregado
    """
    return joblib.load(filepath)


def mlp_preprocess_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Preprocessa DataFrame ja limpo do Telco para treino MLP.

    NAO aplica StandardScaler - deve ser feito APOS o split
    treino/teste para evitar data leakage.

    Args:
        df: DataFrame LIMPO (saida de clean_telco_data())

    Returns:
        Tupla contendo:
            - X: Array de features de shape (n_samples, n_features)
            - y: Array de targets de shape (n_samples,)
            - feature_names: Lista de nomes das features
            - df_encoded: DataFrame com features codificadas

    Raises:
        KeyError: Se a coluna "Churn" nao existir no DataFrame.

    Example:
        >>> import pandas as pd
        >>> from src.data.cleaning import clean_telco_data
        >>> raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        >>> df_clean = clean_telco_data(raw)  # limpeza EDA
        >>> X, y, features, df_enc = mlp_preprocess_data(df_clean)
        >>> print(f"Features: {len(features)}")  # ~45
    """
    df_encoded = encode_target(df)
    df_encoded = one_hot_encode(df_encoded)

    # Separa features e target
    X_df, y_series, feature_names = split_features_target(df_encoded)

    # Converte para arrays numpy
    X = np.asarray(X_df.values, dtype=np.float64)
    y = np.asarray(y_series.values, dtype=np.float64)

    return X, y, feature_names

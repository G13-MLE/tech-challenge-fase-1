"""Limpeza centralizada do dataset Telco (fonte da verdade do EDA).

Este modulo extrai as regras de limpeza e validacao de dominio
descobertas no EDA (Issue #17) para serem reusadas por todos
os pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

MAX_TENURE = 120
"""Limite maximo de tenure segundo validacao de dominio do EDA."""


class DomainValidationError(ValueError):
    """Erro levantado quando anomalias de dominio sao detectadas."""


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza padrao do EDA no dataset Telco.

    Passos (conforme EDA Issue #17):
    1. Valida colunas obrigatorias
    2. Converte TotalCharges para numerico
    3. Remove linhas com TotalCharges NaN
       (clientes recem-assinados, ~0.16% dos dados)
    4. Remove customerID (sem valor preditivo)
    5. Valida anomalias de dominio (tenure, charges)

    Args:
        df: DataFrame bruto do dataset Telco

    Returns:
        DataFrame limpo conforme EDA

    Raises:
        ValueError: Se colunas obrigatorias estiverem ausentes
        DomainValidationError: Se anomalias criticas de dominio
            forem detectadas
    """
    df_clean = df.copy()

    # 1. Valida colunas obrigatorias
    missing = [c for c in REQUIRED_COLUMNS if c not in df_clean.columns]
    if missing:
        msg = f"Colunas obrigatorias ausentes: {missing}"
        raise ValueError(msg)

    # 2. Converte TotalCharges para numerico
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )

    # 3. Remove NaNs em TotalCharges
    # EDA descobriu 11 linhas (~0.16%) com vazio,
    # clientes sem mes faturado
    na_before = len(df_clean)
    df_clean = df_clean.dropna(subset=["TotalCharges"])
    dropped = na_before - len(df_clean)
    if dropped > 0:
        logger.info(
            "[clean_telco_data] Drop de %d registros com "
            "TotalCharges vazio. Shape: %s",
            dropped,
            df_clean.shape,
        )

    # 4. Remove customerID
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # 5. Valida anomalias de dominio
    _validate_domain_anomalies(df_clean)

    return df_clean


def _validate_domain_anomalies(df: pd.DataFrame) -> None:
    """Valida regras de negocio descobertas no EDA.

    Levanta DomainValidationError se anomalias criticas
    forem encontradas.
    """
    anomalies: list[str] = []

    # tenure fora de 0-MAX_TENURE
    tenure_invalid = df[(df["tenure"] < 0) | (df["tenure"] > MAX_TENURE)]
    if len(tenure_invalid) > 0:
        anomalies.append(f"tenure fora de 0-120: {len(tenure_invalid)}")

    # MonthlyCharges negativo
    mc_neg = df[df["MonthlyCharges"] < 0]
    if len(mc_neg) > 0:
        anomalies.append(f"MonthlyCharges negativo: {len(mc_neg)}")

    # TotalCharges negativo
    tc_neg = df[df["TotalCharges"] < 0]
    if len(tc_neg) > 0:
        anomalies.append(f"TotalCharges negativo: {len(tc_neg)}")

    # TotalCharges < MonthlyCharges para tenure > 1
    inconsistent = df[
        (df["tenure"] > 1) & (df["TotalCharges"] < df["MonthlyCharges"])
    ]
    if len(inconsistent) > 0:
        anomalies.append(
            f"TotalCharges < MonthlyCharges para tenure>1: {len(inconsistent)}"
        )

    # Valida colunas categoricas Yes/No
    _validate_yes_no_columns(df, anomalies)

    if anomalies:
        msg = "\n".join(f"- {a}" for a in anomalies)
        raise DomainValidationError(f"Anomalias de dominio detectadas:\n{msg}")


def _validate_yes_no_columns(df: pd.DataFrame, anomalies: list[str]) -> None:
    """Valida que colunas categoricas binarias contem apenas Yes/No."""
    yes_no_cols = [
        c
        for c in [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "Churn",
        ]
        if c in df.columns
    ]

    for col in yes_no_cols:
        invalid = df[~df[col].isin(["Yes", "No"]) & df[col].notna()]
        if len(invalid) > 0:
            anomalies.append(f"{col} com valor fora de Yes/No: {len(invalid)}")


def get_cleaning_summary(
    df_raw: pd.DataFrame, df_clean: pd.DataFrame
) -> dict[str, Any]:
    """Retorna resumo das transformacoes aplicadas pela limpeza.

    Args:
        df_raw: DataFrame original (antes da limpeza)
        df_clean: DataFrame apos clean_telco_data

    Returns:
        Dict com estatisticas de limpeza
    """
    return {
        "rows_before": len(df_raw),
        "rows_after": len(df_clean),
        "rows_dropped": len(df_raw) - len(df_clean),
        "customer_id_removed": "customerID" not in df_clean.columns,
        "totalcharges_numeric": pd.api.types.is_float_dtype(
            df_clean["TotalCharges"]
        )
        if "TotalCharges" in df_clean.columns
        else False,
    }

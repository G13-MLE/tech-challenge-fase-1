"""Testes para o modulo de limpeza centralizada do dataset Telco."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning import (
    DomainValidationError,
    clean_telco_data,
    get_cleaning_summary,
)

ROW_COUNT = 3
DROP_ROW_COUNT = 2
TOTAL_CHARGES_FIRST = 29.85


def _make_valid_df() -> pd.DataFrame:
    """Cria DataFrame minimo valido para clean_telco_data."""
    return pd.DataFrame(
        {
            "customerID": ["c1", "c2", "c3"],
            "tenure": [1, 24, 72],
            "MonthlyCharges": [29.85, 56.95, 89.10],
            "TotalCharges": ["29.85", "1366.80", "6415.20"],
            "Churn": ["No", "Yes", "No"],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "PhoneService": ["No", "Yes", "Yes"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
        }
    )


def test_clean_telco_data_removes_customer_id() -> None:
    """customerID deve ser removido pelo limpeza."""
    df = _make_valid_df()
    result = clean_telco_data(df)
    assert "customerID" not in result.columns


def test_clean_telco_data_converts_total_charges() -> None:
    """TotalCharges deve ser convertido para numerico."""
    df = _make_valid_df()
    result = clean_telco_data(df)
    assert pd.api.types.is_float_dtype(result["TotalCharges"])
    assert result["TotalCharges"].iloc[0] == TOTAL_CHARGES_FIRST


def test_clean_telco_data_drops_empty_total_charges() -> None:
    """Linhas com TotalCharges vazio devem ser removidas."""
    df = _make_valid_df()
    df.loc[1, "TotalCharges"] = " "
    result = clean_telco_data(df)
    assert len(result) == DROP_ROW_COUNT
    assert "c2" not in result.get("customerID", pd.Series())


def test_clean_telco_data_raises_for_missing_columns() -> None:
    """Deve levantar ValueError se colunas obrigatorias faltam."""
    df = pd.DataFrame({"customerID": ["c1"], "Churn": ["No"]})
    with pytest.raises(ValueError, match="Colunas obrigatorias ausentes"):
        clean_telco_data(df)


def test_clean_telco_data_raises_for_negative_tenure() -> None:
    """Deve levantar DomainValidationError para tenure negativo."""
    df = _make_valid_df()
    df.loc[0, "tenure"] = -1
    with pytest.raises(DomainValidationError, match="tenure fora de 0-120"):
        clean_telco_data(df)


def test_clean_telco_data_raises_for_negative_monthly_charges() -> None:
    """Deve levantar DomainValidationError para MonthlyCharges negativo."""
    df = _make_valid_df()
    df.loc[0, "MonthlyCharges"] = -5.0
    with pytest.raises(DomainValidationError, match="MonthlyCharges negativo"):
        clean_telco_data(df)


def test_clean_telco_data_raises_for_inconsistent_charges() -> None:
    """Deve levantar DomainValidationError quando TotalCharges <
    MonthlyCharges para tenure > 1."""
    df = _make_valid_df()
    df.loc[1, "tenure"] = 5
    df.loc[1, "TotalCharges"] = "10.0"
    df.loc[1, "MonthlyCharges"] = 50.0
    with pytest.raises(
        DomainValidationError,
        match="TotalCharges < MonthlyCharges",
    ):
        clean_telco_data(df)


def test_clean_telco_data_raises_for_invalid_churn_values() -> None:
    """Deve levantar DomainValidationError para valores invalidos em
    colunas Yes/No."""
    df = _make_valid_df()
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(
        DomainValidationError, match="Churn com valor fora de Yes/No"
    ):
        clean_telco_data(df)


def test_get_cleaning_summary() -> None:
    """Resumo deve refletir transformacoes aplicadas."""
    df_raw = _make_valid_df()
    df_clean = clean_telco_data(df_raw)
    summary = get_cleaning_summary(df_raw, df_clean)

    assert summary["rows_before"] == ROW_COUNT
    assert summary["rows_after"] == ROW_COUNT
    assert summary["rows_dropped"] == 0
    assert summary["customer_id_removed"] is True
    assert summary["totalcharges_numeric"] is True

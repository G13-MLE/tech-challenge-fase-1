"""Testes para carregamento e limpeza do dataset Telco."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.data.load import load_telco_data


def test_load_telco_data_success() -> None:
    """Teste de carregamento com apply_cleaning=False (raw)."""
    mock_df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "tenure": [1, 24],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", "56.95"],
            "Churn": ["Yes", "No"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "PhoneService": ["No", "Yes"],
            "PaperlessBilling": ["Yes", "No"],
        }
    )

    with patch(
        "src.data.load.pd.read_csv", return_value=mock_df
    ) as mock_read_csv:
        result = load_telco_data("dummy/path.csv", apply_cleaning=False)

        mock_read_csv.assert_called_once_with("dummy/path.csv")
        pd.testing.assert_frame_equal(result, mock_df)


def test_load_telco_data_file_not_found() -> None:
    """Teste de excecao quando arquivo nao existe."""
    with (
        patch(
            "src.data.load.pd.read_csv",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(FileNotFoundError, match="nao encontrado"),
    ):
        load_telco_data("missing_file.csv")


def test_load_telco_data_default_applies_cleaning() -> None:
    """Por padrao, load_telco_data deve aplicar clean_telco_data."""
    mock_df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "tenure": [1, 24],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", "683.40"],
            "Churn": ["Yes", "No"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "PhoneService": ["No", "Yes"],
            "PaperlessBilling": ["Yes", "No"],
        }
    )

    with patch(
        "src.data.load.pd.read_csv", return_value=mock_df
    ) as mock_read_csv:
        result = load_telco_data("dummy/path.csv")

        mock_read_csv.assert_called_once_with("dummy/path.csv")
        # customerID deve ter sido removido pela limpeza
        assert "customerID" not in result.columns
        # TotalCharges deve ser numerico
        assert pd.api.types.is_float_dtype(result["TotalCharges"])

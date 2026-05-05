"""Tests for Pandera schema validation of Telco dataset.

Validates that the schema correctly accepts valid data and
rejects invalid data with appropriate errors.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.schemas.pandera_schemas import TELCO_RAW_SCHEMA

_NUM_ROWS = 2  # number of rows in test fixture


def _make_valid_df() -> pd.DataFrame:
    """Cria DataFrame minimo que satisfaz o schema bruto Telco."""
    return pd.DataFrame(
        {
            "customerID": ["0001-X", "0002-Y"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 34],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No internet service"],
            "OnlineBackup": ["No", "No internet service"],
            "DeviceProtection": ["No internet service", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No internet service", "Yes"],
            "StreamingMovies": ["No", "No internet service"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", "1889.50"],
            "Churn": ["No", "Yes"],
        }
    )


class TestTelcoRawSchemaValidData:
    """Schema aceita dados validos."""

    def test_valid_dataframe_passes(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        result = TELCO_RAW_SCHEMA.validate(df)
        assert len(result) == _NUM_ROWS

    def test_valid_dataframe_preserves_columns(  # noqa: PLR6301
        self,
    ) -> None:
        df = _make_valid_df()
        result = TELCO_RAW_SCHEMA.validate(df)
        assert set(result.columns) == set(TELCO_RAW_SCHEMA.columns.keys())


class TestTelcoRawSchemaInvalidValues:
    """Schema rejeita valores invalidos de dominio."""

    def test_invalid_gender_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df.loc[0, "gender"] = "Other"
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_invalid_senior_citizen_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df.loc[0, "SeniorCitizen"] = 5
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_invalid_tenure_negative_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df.loc[0, "tenure"] = -1
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_invalid_tenure_over_120_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df.loc[0, "tenure"] = 200
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_invalid_churn_value_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df.loc[0, "Churn"] = "Maybe"
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_invalid_contract_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df.loc[0, "Contract"] = "Weekly"
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_negative_monthly_charges_rejected(  # noqa: PLR6301
        self,
    ) -> None:
        df = _make_valid_df()
        df.loc[0, "MonthlyCharges"] = -10.0
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)


class TestTelcoRawSchemaMissingColumns:
    """Schema rejeita DataFrames com colunas ausentes (strict=True)."""

    def test_missing_column_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df().drop(columns=["Churn"])
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_extra_column_rejected(self) -> None:  # noqa: PLR6301
        df = _make_valid_df()
        df["ExtraColumn"] = "invalid"
        with pytest.raises((SchemaError, SchemaErrors)):
            TELCO_RAW_SCHEMA.validate(df)

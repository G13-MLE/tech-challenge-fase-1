"""Schemas de validacao Pandera para o dataset Telco Customer Churn.

Define contratos de dados para o dataset bruto e processado,
garantindo integridade de colunas, tipos e dominios de valor
em tempo de execucao.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# Valores validos para colunas categoricas do dataset Telco
_YES_NO = ["Yes", "No"]
_GENDER = ["Female", "Male"]
_INTERNET = ["DSL", "Fiber optic", "No"]
_MULTIPLE_LINES = ["Yes", "No", "No phone service"]
_INTERNET_ADDON = ["Yes", "No", "No internet service"]
_CONTRACT = ["Month-to-month", "One year", "Two year"]
_PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

TELCO_RAW_SCHEMA: DataFrameSchema = DataFrameSchema(
    name="TelcoRawSchema",
    columns={
        "customerID": Column(
            dtype="object",
            nullable=False,
            description="Identificador unico do cliente",
        ),
        "gender": Column(
            dtype="object",
            checks=Check.isin(_GENDER),
            nullable=False,
            description="Genero do cliente",
        ),
        "SeniorCitizen": Column(
            dtype="int64",
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Indica se e idoso (0/1)",
        ),
        "Partner": Column(
            dtype="object",
            checks=Check.isin(_YES_NO),
            nullable=False,
            description="Possui parceiro(a)",
        ),
        "Dependents": Column(
            dtype="object",
            checks=Check.isin(_YES_NO),
            nullable=False,
            description="Possui dependentes",
        ),
        "tenure": Column(
            dtype="int64",
            checks=Check.in_range(0, 120),
            nullable=False,
            description="Meses de permanencia",
        ),
        "PhoneService": Column(
            dtype="object",
            checks=Check.isin(_YES_NO),
            nullable=False,
            description="Servico de telefone",
        ),
        "MultipleLines": Column(
            dtype="object",
            checks=Check.isin(_MULTIPLE_LINES),
            nullable=False,
            description="Multiplas linhas",
        ),
        "InternetService": Column(
            dtype="object",
            checks=Check.isin(_INTERNET),
            nullable=False,
            description="Tipo de internet",
        ),
        "OnlineSecurity": Column(
            dtype="object",
            checks=Check.isin(_INTERNET_ADDON),
            nullable=False,
            description="Seguranca online",
        ),
        "OnlineBackup": Column(
            dtype="object",
            checks=Check.isin(_INTERNET_ADDON),
            nullable=False,
            description="Backup online",
        ),
        "DeviceProtection": Column(
            dtype="object",
            checks=Check.isin(_INTERNET_ADDON),
            nullable=False,
            description="Protecao de dispositivo",
        ),
        "TechSupport": Column(
            dtype="object",
            checks=Check.isin(_INTERNET_ADDON),
            nullable=False,
            description="Suporte tecnico",
        ),
        "StreamingTV": Column(
            dtype="object",
            checks=Check.isin(_INTERNET_ADDON),
            nullable=False,
            description="Streaming de TV",
        ),
        "StreamingMovies": Column(
            dtype="object",
            checks=Check.isin(_INTERNET_ADDON),
            nullable=False,
            description="Streaming de filmes",
        ),
        "Contract": Column(
            dtype="object",
            checks=Check.isin(_CONTRACT),
            nullable=False,
            description="Tipo de contrato",
        ),
        "PaperlessBilling": Column(
            dtype="object",
            checks=Check.isin(_YES_NO),
            nullable=False,
            description="Fatura sem papel",
        ),
        "PaymentMethod": Column(
            dtype="object",
            checks=Check.isin(_PAYMENT),
            nullable=False,
            description="Metodo de pagamento",
        ),
        "MonthlyCharges": Column(
            dtype="float64",
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Cobranca mensal",
        ),
        "TotalCharges": Column(
            dtype="object",
            nullable=True,
            description="Cobranca total (string, pode ter vazios)",
        ),
        "Churn": Column(
            dtype="object",
            checks=Check.isin(_YES_NO),
            nullable=False,
            description="Cancelamento (variavel alvo)",
        ),
    },
    strict=True,
    description="Schema do dataset Telco Customer Churn bruto (IBM)",
)


def validate_telco_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida DataFrame contra o schema bruto do Telco.

    Args:
        df: DataFrame bruto do dataset Telco

    Returns:
        DataFrame validado (coerced se necessario)

    Raises:
        SchemaError: Se a validacao falhar
    """
    return TELCO_RAW_SCHEMA.validate(df)

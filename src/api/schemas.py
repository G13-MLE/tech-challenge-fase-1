from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, populate_by_name=True
    )

    customer_id: str = Field(
        ..., alias="customerID", description="ID único do cliente"
    )
    tenure: int = Field(
        ..., description="Número de meses que o cliente permaneceu na empresa"
    )
    monthly_charges: float = Field(
        ...,
        alias="MonthlyCharges",
        description="Valor cobrado mensalmente do cliente",
    )
    contract: str = Field(
        ...,
        alias="Contract",
        description="Tipo de contrato (ex: Month-to-month)",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    churn_probability: float = Field(
        ...,
        description="Probabilidade do cliente cancelar o serviço (0.0 a 1.0)",
    )
    churn_prediction: bool = Field(
        ..., description="Predição binária de churn (True/False)"
    )

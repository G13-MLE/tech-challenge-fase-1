from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customerID: str = Field(..., description="ID único do cliente")
    tenure: int = Field(
        ..., description="Número de meses que o cliente permaneceu na empresa"
    )
    MonthlyCharges: float = Field(
        ..., description="Valor cobrado mensalmente do cliente"
    )
    Contract: str = Field(
        ...,
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

from __future__ import annotations

from fastapi import FastAPI

from src.api.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn de clientes da Telco",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Verifica se a API está no ar."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Recebe os dados do cliente e retorna a predição de churn."""
    # TODO: Integrar com o modelo real no futuro
    # Por enquanto, retornamos um mock dinâmico baseado na entrada
    tenure_threshold = 12
    churn_threshold = 0.5

    probability = 0.85 if request.tenure < tenure_threshold else 0.15
    prediction = probability > churn_threshold

    return PredictResponse(
        churn_probability=probability,
        churn_prediction=prediction,
    )

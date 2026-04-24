"""Churn Prediction API."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import LatencyMiddleware, RequestIDMiddleware
from src.api.schemas import PredictRequest, PredictResponse
from src.logging_config import LoggingConfig, request_id_ctx, setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    """Application lifespan: setup logging on startup."""
    setup_logging(LoggingConfig(json_format=True))
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="API para predicao de churn de clientes da Telco",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware registration: RequestIDMiddleware added LAST so it
# runs FIRST on request (inbound) and LAST on response (outbound)
# in LIFO order of FastAPI. This ensures request_id is set when
# LatencyMiddleware logs on response completion.
app.add_middleware(LatencyMiddleware)
app.add_middleware(RequestIDMiddleware)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Verifica se a API esta no ar."""
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
)
async def predict(
    request: PredictRequest,
) -> PredictResponse:
    """Recebe os dados do cliente e retorna a predicao de churn."""
    start = time.perf_counter()

    # TODO: Integrar com o modelo real no futuro
    # Por enquanto, retornamos um mock dinamico baseado na entrada
    tenure_threshold = 12
    churn_threshold = 0.5

    probability = 0.85 if request.tenure < tenure_threshold else 0.15
    prediction = probability > churn_threshold

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Prediction completed: %s",
        {
            "customer_id": request.customer_id,
            "tenure": request.tenure,
            "request_id": request_id_ctx.get(""),
            "prediction_latency_ms": round(elapsed_ms, 2),
            "churn_prediction": prediction,
            "churn_probability": probability,
        },
    )

    return PredictResponse(
        churn_probability=probability,
        churn_prediction=prediction,
    )

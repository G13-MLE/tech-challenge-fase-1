"""API de Predição de Churn."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from src.api.logging import LoggingConfig, request_id_ctx, setup_logging
from src.api.metrics import PREDICTION_PROBABILITY, metrics_exposition
from src.api.middleware import (
    DriftMiddleware,
    LatencyMiddleware,
    RequestIDMiddleware,
)
from src.api.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    """Ciclo de vida da aplicação: configura o logging na inicialização."""
    setup_logging(LoggingConfig(json_format=True))
    yield


app = FastAPI(
    title="API de Predição de Churn",
    description="API para predição de churn de clientes da Telco",
    version="0.1.0",
    lifespan=lifespan,
)

# Registro de middleware (ordem LIFO do FastAPI):
# DriftMiddleware adicionado POR ÚLTIMO para executar PRIMEIRO na entrada,
# garantindo que o body seja lido e reconstruído antes dos demais.
# RequestIDMiddleware é o último na saída (primeiro na entrada) para
# garantir que o request_id esteja definido quando LatencyMiddleware
# registrar o log ao completar a resposta.
app.add_middleware(LatencyMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(DriftMiddleware)


@app.get("/health", tags=["Saúde"])
async def health_check() -> dict[str, str]:
    """Verifica se a API está no ar."""
    return {"status": "healthy"}


@app.get("/metrics", tags=["Monitoramento"])
async def metrics() -> Response:
    """Expoem métricas no formato Prometheus.

    Endpoint compatível com scrapers como Prometheus Server
    ou Grafana Agent para coleta de telemetria.
    """
    return Response(
        content=metrics_exposition(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predição"],
)
async def predict(
    request: PredictRequest,
) -> PredictResponse:
    """Recebe os dados do cliente e retorna a predição de churn."""
    start = time.perf_counter()

    # TODO: Integrar com o modelo real no futuro
    # Por enquanto, retornamos um mock dinâmico baseado na entrada
    tenure_threshold = 12
    churn_threshold = 0.5

    probability = 0.85 if request.tenure < tenure_threshold else 0.15
    prediction = probability > churn_threshold

    elapsed_ms = (time.perf_counter() - start) * 1000

    PREDICTION_PROBABILITY.observe(probability)

    logger.info(
        "Predição concluída: %s",
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

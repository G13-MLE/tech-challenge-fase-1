"""Middleware para detecção de data drift em requisições /predict.

Intercepta requisições POST /predict, lê o body JSON, detecta drift
comparando contra a baseline de treino e registra métricas Prometheus.
Reconstrói o stream do body para que o endpoint possa consumi-lo
normalmente.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, override

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.api.drift import detect_drift
from src.api.metrics import DRIFT_DETECTIONS_TOTAL

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

# Features monitoradas para drift (devem bater com PredictRequest)
_DRIFT_FEATURES = ["tenure", "MonthlyCharges", "Contract"]


class DriftMiddleware(BaseHTTPMiddleware):
    """Middleware ASGI que detecta data drift nas requisições /predict.

    Lê o body JSON da requisição, reconstrói o stream para o endpoint
    e compara as features contra a baseline de treinamento.
    """

    @override
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method != "POST" or request.url.path != "/predict":
            return await call_next(request)

        body = await request.body()
        request._receive = DriftMiddleware._build_receive(body)  # type: ignore[attr-defined,assignment]

        DriftMiddleware._detect_and_log(body)

        return await call_next(request)

    @staticmethod
    def _build_receive(body: bytes) -> object:
        """Reconstrói o ASGI receive para que o endpoint possa ler o body."""

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": body, "more_body": False}

        return receive

    @staticmethod
    def _detect_and_log(body: bytes) -> None:
        """Parseia o body, detecta drift e registra métricas/logs."""
        try:
            data = json.loads(body)
            features = {f: data.get(f) for f in _DRIFT_FEATURES if f in data}
            if not features:
                return

            drift_report = detect_drift(features)

            for feature_name, feature_info in drift_report.features.items():
                DRIFT_DETECTIONS_TOTAL.labels(
                    feature=feature_name,
                    drift_detected=str(feature_info["score"] > 0.0).lower(),
                ).inc()

            if drift_report.drift_detected:
                logger.warning(
                    "Data drift detectado: score=%s features=%s",
                    drift_report.drift_score,
                    drift_report.features,
                )
            else:
                logger.debug(
                    "Data drift: estável (score=%s)", drift_report.drift_score
                )

        except (json.JSONDecodeError, ValueError):
            logger.debug("Drift middleware: body não é JSON válido")

"""Middleware para detecção de data drift em requisições /predict.

Intercepta requisições POST /predict, lê o body JSON, detecta drift
comparando contra a baseline de treino e registra métricas Prometheus.
Reconstrói o stream do body para que o endpoint possa consumi-lo
normalmente.

Alem da deteccao per-request (out-of-range / categoria inedita),
acumula amostras numericas em buffer circular e periodicamente
calcula PSI real comparando a janela contra a baseline de treino.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, override

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.api.drift import _load_reference_stats, detect_drift
from src.api.drift_monitor import PsiResult, PsiWindow
from src.api.metrics import DRIFT_DETECTIONS_TOTAL, DRIFT_PSI_GAUGE

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

# Features monitoradas para drift (devem bater com PredictRequest)
_DRIFT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Features numericas acumuladas em janela para calculo de PSI
_NUMERIC_FEATURES = frozenset(["tenure", "MonthlyCharges", "TotalCharges"])

# Janelas de PSI por feature numerica
_psi_windows: dict[str, PsiWindow] = {}

# Contador de requisicoes para disparo periodico do PSI
_sample_state: dict[str, int] = {"count": 0}

# A cada N requisicoes, recalcula PSI das janelas
_PSI_EVERY_N = 50


def _compute_window_psi() -> None:
    """Calcula PSI para todas as janelas prontas e registra no gauge."""
    try:
        reference = _load_reference_stats()["features"]
    except (FileNotFoundError, json.JSONDecodeError):
        return

    for feature_name, window in _psi_windows.items():
        if not window.ready:
            continue

        baseline = reference.get(feature_name)
        if baseline is None or baseline["type"] != "numeric":
            continue

        result = PsiResult.from_window(window, baseline["bins"])
        DRIFT_PSI_GAUGE.labels(feature=feature_name).set(result.score)

        if result.status != "stable":
            logger.warning(
                "PSI %s para feature '%s': score=%.4f",
                result.status,
                result.feature,
                result.score,
            )

        window.reset()


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
        """Parseia o body, detecta drift e registra metricas/logs.

        Alem da deteccao per-request, alimenta janelas de PSI para
        features numericas e periodicamente recalcula o indice.
        """
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

            # Alimenta janelas de PSI para features numericas
            for feature_name in _NUMERIC_FEATURES:
                value = features.get(feature_name)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue

                if feature_name not in _psi_windows:
                    _psi_windows[feature_name] = PsiWindow.new(feature_name)
                _psi_windows[feature_name].add(numeric_value)

            _sample_state["count"] += 1
            if _sample_state["count"] % _PSI_EVERY_N == 0:
                _compute_window_psi()

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

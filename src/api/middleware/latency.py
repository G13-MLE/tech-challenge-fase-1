"""Middleware para medição de latência por requisição.

Mede o tempo de wall-clock para cada requisição usando
`time.perf_counter()` e registra o resultado como JSON estruturado.
Emite WARNING quando a latência ultrapassa o limite de SLO.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.api.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
)

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

_DEFAULT_SLO_MS = 500.0


class LatencyMiddleware(BaseHTTPMiddleware):
    """Middleware ASGI que mede a latência de requisições.

    Registra um registro estruturado para cada requisição contendo:
    - method, path, status_code
    - latency_ms (milisegundos de wall-clock)
    - slo_ms (limiar)
    - slo_breached (boolean)

    Emite INFO para requisições dentro do SLO, WARNING para violações.
    """

    def __init__(
        self,
        app: object,
        slo_ms: float | None = None,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.slo_ms = (
            slo_ms
            if slo_ms is not None
            else float(os.getenv("PREDICTION_SLO_MS", str(_DEFAULT_SLO_MS)))
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_s = time.perf_counter() - start
        elapsed_ms = elapsed_s * 1000

        path = request.url.path
        method = request.method
        status_code = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            status_code=status_code,
            path=path,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            path=path,
        ).observe(elapsed_s)

        slo_breached = elapsed_ms > self.slo_ms

        log_data = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(elapsed_ms, 2),
            "slo_ms": self.slo_ms,
            "slo_breached": slo_breached,
        }

        if slo_breached:
            logger.warning("Violação de SLO: %s", log_data)
        else:
            logger.info("Requisição concluída: %s", log_data)

        return response

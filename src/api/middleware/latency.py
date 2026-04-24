"""Middleware for per-request latency measurement.

Measures wall-clock time for each request using
time.perf_counter() and logs the result as structured JSON.
Emits WARNING when latency exceeds the SLO threshold.
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

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

_DEFAULT_SLO_MS = 500.0


class LatencyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that measures request latency.

    Logs a structured record for every request containing:
    - method, path, status_code
    - latency_ms (wall-clock milliseconds)
    - slo_ms (threshold)
    - slo_breached (boolean)

    Emits INFO for requests within SLO, WARNING for breaches.
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
        elapsed_ms = (time.perf_counter() - start) * 1000

        slo_breached = elapsed_ms > self.slo_ms

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "slo_ms": self.slo_ms,
            "slo_breached": slo_breached,
        }

        if slo_breached:
            logger.warning("SLO breach: %s", log_data)
        else:
            logger.info("Request completed: %s", log_data)

        return response

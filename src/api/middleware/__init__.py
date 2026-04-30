"""Pacote de middlewares da API.

Re-exporta as classes de middleware para importação conveniente:
    from src.api.middleware import (
        RequestIDMiddleware, LatencyMiddleware,
    )
"""

from __future__ import annotations

from src.api.middleware.latency import LatencyMiddleware
from src.api.middleware.request_id import RequestIDMiddleware

__all__ = ["LatencyMiddleware", "RequestIDMiddleware"]

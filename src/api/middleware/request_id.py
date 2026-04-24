"""Middleware for request ID generation and propagation.

Generates a UUID4 request ID for every incoming request that
does not already carry an X-Request-ID header. Propagates the
ID to both the response header and the logging ContextVar.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, override

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.logging_config import request_id_ctx

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

HEADER_NAME = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that assigns a request ID to each request.

    If the incoming request already carries an X-Request-ID
    header, that value is reused (supports distributed tracing).
    Otherwise, a new UUID4 is generated. The ID is set on:
    - The logging ContextVar (available to all log records)
    - The response X-Request-ID header
    """

    @override
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get(HEADER_NAME, str(uuid.uuid4()))

        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers[HEADER_NAME] = request_id
            return response
        finally:
            request_id_ctx.reset(token)

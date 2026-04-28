"""Middleware para geração e propagação de ID de requisição.

Gera um UUID4 como request ID para cada requisição recebida que
ainda não possua o cabeçalho X-Request-ID. Propaga o
ID tanto no cabeçalho da resposta quanto na ContextVar de logging.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, override

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.api.logging import request_id_ctx

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

HEADER_NAME = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware ASGI que atribui um request ID a cada requisição.

    Se a requisição já possuir um cabeçalho X-Request-ID,
    esse valor é reutilizado (suporta tracing distribuído).
    Caso contrário, um novo UUID4 é gerado. O ID é definido em:
    - A ContextVar de logging (disponível em todos os registros de log)
    - O cabeçalho X-Request-ID da resposta
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

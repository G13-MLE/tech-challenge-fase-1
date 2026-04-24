"""Configuração centralizada de logging para saída JSON estruturada.

Fornece LoggingConfig, setup_logging(), RequestContextFilter,
e a ContextVar request_id_ctx para rastreamento de requisições
cross-cutting entre scripts de pipeline e a aplicação FastAPI.
"""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pythonjsonlogger.json import JsonFormatter

if TYPE_CHECKING:
    from logging import Formatter

# ContextVar para rastreamento de requisições entre registros de log
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


@dataclass(frozen=True)
class LoggingConfig:
    """Configuração para setup de logging estruturado.

    Atributos:
        level: String do nível de log (DEBUG, INFO, WARNING, ERROR).
        json_format: Se True, emite logs JSON; caso contrário,
            formato de texto legível para desenvolvimento local.
        slo_ms: Limiar de latência SLO em milisegundos. Logs
            que excederem emitem WARNING em vez de INFO.
    """

    level: str = "INFO"
    json_format: bool = True
    slo_ms: float = 500.0


class RequestContextFilter(logging.Filter):
    """Injeta request_id em todo registro de log.

    Lê o valor atual de request_id_ctx e o adiciona ao
    LogRecord como um atributo. Isso permite que formatadores
    JSON e de texto incluam request_id sem acoplar os locais
    de chamada de log à variável de contexto.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("")  # type: ignore[attr-defined]
        return True


def _json_formatter() -> Formatter:
    """Cria um formatador JSON com campos padrão.

    Campos incluídos: timestamp, level, logger,
    message, request_id. Quaisquer kwargs extras passados
    para chamadas de log aparecem como chaves adicionais.
    """
    fmt = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        },
    )
    return fmt


def _text_formatter() -> Formatter:
    """Cria um formatador de texto legível para desenvolvimento local.

    Inclui request_id quando presente; caso contrário,
    saída compacta adequada para leitura no terminal.
    """
    return logging.Formatter(
        fmt=(
            "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configura o root logger para saída estruturada.

    Configura o root logger com um RequestContextFilter
    e um formatador JSON ou de texto baseado na configuração.
    Seguro para chamar múltiplas vezes (idempotente em chamadas repetidas).

    Args:
        config: Instância de LoggingConfig. Se None, lê das
            variáveis de ambiente:
            - LOG_LEVEL (padrão: INFO)
            - LOG_FORMAT (padrão: json, aceita text)
            - PREDICTION_SLO_MS (padrão: 500.0)
    """
    if config is None:
        config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            json_format=(os.getenv("LOG_FORMAT", "json").lower() == "json"),
            slo_ms=float(os.getenv("PREDICTION_SLO_MS", "500.0")),
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))

    # Remove handlers existentes para evitar saída duplicada
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    if config.json_format:
        handler.setFormatter(_json_formatter())
    else:
        handler.setFormatter(_text_formatter())

    # Anexa o filtro de contexto de requisição a todos os handlers
    handler.addFilter(RequestContextFilter())

    root_logger.addHandler(handler)

    # Silencia loggers de terceiros excessivamente barulhentos
    for noisy in ("urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

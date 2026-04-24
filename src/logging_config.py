"""Centralized logging configuration for structured JSON output.

Provides LoggingConfig, setup_logging(), RequestContextFilter,
and the request_id_ctx ContextVar for cross-cutting request
tracing across pipeline scripts and the FastAPI application.
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

# ContextVar for request tracing across log records
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for structured logging setup.

    Attributes:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, emit JSON logs; else human-readable
            text format suitable for local development.
        slo_ms: Latency SLO threshold in milliseconds. Logs
            exceeding this emit a WARNING instead of INFO.
    """

    level: str = "INFO"
    json_format: bool = True
    slo_ms: float = 500.0


class RequestContextFilter(logging.Filter):
    """Injects request_id into every log record.

    Reads the current value of request_id_ctx and adds it
    to the LogRecord as an attribute. This allows JSON
    formatters and text formatters to include request_id
    without coupling log call sites to the context variable.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("")  # type: ignore[attr-defined]
        return True


def _json_formatter() -> Formatter:
    """Create a JSON formatter with standard fields.

    Fields included: timestamp, level, logger,
    message, request_id. Any extra kwargs passed to
    log calls appear as additional top-level keys.
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
    """Create a human-readable text formatter for local dev.

    Includes request_id when present; otherwise compact
    output suitable for terminal reading.
    """
    return logging.Formatter(
        fmt=(
            "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure the root logger for structured output.

    Sets up the root logger with a RequestContextFilter
    and either a JSON or text formatter based on config.
    Safe to call multiple times (idempotent on repeat calls).

    Args:
        config: LoggingConfig instance. If None, reads from
            environment variables:
            - LOG_LEVEL (default: INFO)
            - LOG_FORMAT (default: json, accepts text)
            - PREDICTION_SLO_MS (default: 500.0)
    """
    if config is None:
        config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            json_format=(os.getenv("LOG_FORMAT", "json").lower() == "json"),
            slo_ms=float(os.getenv("PREDICTION_SLO_MS", "500.0")),
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))

    # Remove existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    if config.json_format:
        handler.setFormatter(_json_formatter())
    else:
        handler.setFormatter(_text_formatter())

    # Attach request context filter to all handlers
    handler.addFilter(RequestContextFilter())

    root_logger.addHandler(handler)

    # Silence overly noisy third-party loggers
    for noisy in ("urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

"""Testes para o módulo src.api.logging."""

from __future__ import annotations

import logging

from src.api.logging import (
    LoggingConfig,
    RequestContextFilter,
    request_id_ctx,
    setup_logging,
)

_DEFAULT_SLO_MS = 500.0


def test_logging_config_defaults() -> None:
    """LoggingConfig deve ter defaults sensatos."""
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.json_format is True
    assert config.slo_ms == _DEFAULT_SLO_MS


def test_request_context_filter_injects_request_id() -> None:
    """RequestContextFilter deve adicionar request_id aos registros."""
    token = request_id_ctx.set("test-req-123")
    try:
        f = RequestContextFilter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        f.filter(record)
        assert record.request_id == "test-req-123"  # type: ignore[attr-defined]
    finally:
        request_id_ctx.reset(token)


def test_request_context_filter_default_empty() -> None:
    """RequestContextFilter usa string vazia quando ID nao estiver definido."""
    token = request_id_ctx.set("")
    try:
        f = RequestContextFilter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        f.filter(record)
        assert not record.request_id  # type: ignore[attr-defined]
    finally:
        request_id_ctx.reset(token)


def test_setup_logging_json_format(monkeypatch: object) -> None:
    """setup_logging com json_format deve usar formatador JSON."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "json")
    setup_logging()

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert "Json" in type(handler.formatter).__name__


def test_setup_logging_text_format(monkeypatch: object) -> None:
    """setup_logging com formato text deve usar formatador padrão."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "text")
    setup_logging()

    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert type(handler.formatter).__name__ == "Formatter"


def test_setup_logging_idempotent() -> None:
    """Chamadas repetidas de setup_logging não devem duplicar handlers."""
    setup_logging(LoggingConfig())
    setup_logging(LoggingConfig())
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_setup_logging_respects_level() -> None:
    """setup_logging com config explícita deve respeitar o level."""
    config = LoggingConfig(level="WARNING", json_format=False)
    setup_logging(config)
    root = logging.getLogger()
    assert root.level == logging.WARNING

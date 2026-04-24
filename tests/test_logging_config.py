"""Tests for src.logging_config module."""

from __future__ import annotations

import logging

from src.logging_config import (
    LoggingConfig,
    RequestContextFilter,
    request_id_ctx,
    setup_logging,
)

_DEFAULT_SLO_MS = 500.0


def test_logging_config_defaults() -> None:
    """LoggingConfig must have sensible defaults."""
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.json_format is True
    assert config.slo_ms == _DEFAULT_SLO_MS


def test_request_context_filter_injects_request_id() -> None:
    """RequestContextFilter must add request_id to records."""
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
    """RequestContextFilter must use empty string when no ID set."""
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
    """setup_logging with json_format must use JSON formatter."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "json")
    setup_logging()

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert "Json" in type(handler.formatter).__name__


def test_setup_logging_text_format(monkeypatch: object) -> None:
    """setup_logging with text format must use standard formatter."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "text")
    setup_logging()

    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert type(handler.formatter).__name__ == "Formatter"


def test_setup_logging_idempotent() -> None:
    """Repeated setup_logging calls must not duplicate handlers."""
    setup_logging(LoggingConfig())
    setup_logging(LoggingConfig())
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_setup_logging_respects_level() -> None:
    """setup_logging with explicit config must honor level."""
    config = LoggingConfig(level="WARNING", json_format=False)
    setup_logging(config)
    root = logging.getLogger()
    assert root.level == logging.WARNING

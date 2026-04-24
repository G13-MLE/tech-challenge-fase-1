"""Tests for API middleware (request ID and latency)."""

from __future__ import annotations

import logging
import uuid

from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app
from src.logging_config import LoggingConfig, setup_logging

_HIGH_CHURN_PROB = 0.85

setup_logging(LoggingConfig(json_format=False))

client = TestClient(app)


def test_health_endpoint_still_works() -> None:
    """Health endpoint must work after middleware is added."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_still_works() -> None:
    """Predict endpoint must work after middleware is added."""
    payload = {
        "customerID": "7590-VHVEG",
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "Contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["churn_probability"] == _HIGH_CHURN_PROB
    assert data["churn_prediction"] is True


def test_request_id_returned_in_response_header() -> None:
    """Response must include X-Request-ID header."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert "X-Request-ID" in response.headers
    uuid.UUID(response.headers["X-Request-ID"])


def test_request_id_propagated_from_client() -> None:
    """Client-supplied X-Request-ID must be preserved."""
    custom_id = "my-custom-trace-id-123"
    response = client.get(
        "/health",
        headers={"X-Request-ID": custom_id},
    )
    assert response.headers["X-Request-ID"] == custom_id


def test_latency_middleware_logs_on_request(caplog: object) -> None:
    """LatencyMiddleware must log request completion."""
    with caplog.at_level(logging.INFO):  # type: ignore[union-attr]
        response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert any(
        "Request completed" in r.message or "SLO breach" in r.message
        for r in caplog.records  # type: ignore[union-attr]
    )

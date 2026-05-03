"""Testes para exposicao e coleta de metricas Prometheus."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.metrics import DRIFT_PSI_GAUGE
from tests.test_api import base_payload

client = TestClient(app)


def test_metrics_endpoint_returns_prometheus_text() -> None:
    """/metrics deve retornar conteúdo no formato Prometheus."""
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "text/plain" in response.headers["content-type"]


def test_http_requests_total_present() -> None:
    """O contador http_requests_total deve aparecer nas métricas."""
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "http_requests_total{" in response.text
    assert 'method="GET"' in response.text
    assert 'path="/health"' in response.text


def test_http_request_duration_seconds_present() -> None:
    """O histograma de latência deve aparecer nas métricas expostas."""
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "http_request_duration_seconds_bucket" in response.text
    assert 'method="GET",path="/health"' in response.text


@patch("src.api.main.predict_single", return_value=0.85)
def test_prediction_probability_histogram_populated(
    mock_predict: object,
) -> None:
    """O histograma de probabilidade deve registrar valores após /predict."""
    payload = base_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "prediction_probability_bucket" in response.text
    assert 'le="0.9"' in response.text
    assert "prediction_probability_count" in response.text


def test_drift_psi_gauge_present() -> None:
    """O gauge drift_psi_score deve aparecer nas metricas expostas."""
    DRIFT_PSI_GAUGE.labels(feature="tenure").set(0.05)

    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "drift_psi_score" in response.text
    assert 'feature="tenure"' in response.text

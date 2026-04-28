"""Testes para exposição e coleta de métricas Prometheus."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app

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


def test_prediction_probability_histogram_populated() -> None:
    """O histograma de probabilidade deve registrar valores após /predict."""
    payload = {
        "customerID": "7590-VHVEG",
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "Contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "prediction_probability_bucket" in response.text
    assert 'le="0.9"' in response.text
    assert "prediction_probability_count" in response.text

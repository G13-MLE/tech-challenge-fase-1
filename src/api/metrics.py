"""Métricas Prometheus para monitoramento da API.

Expõe contadores, histogramas e sumários no formato do Prometheus
para consumo por scrapers (ex: Grafana Agent, Prometheus Server).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest

# Contador de requisições HTTP por método, status_code e path
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["method", "status_code", "path"],
)

# Histograma de latência de requisições HTTP (em segundos)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP em segundos",
    ["method", "path"],
    buckets=[
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ],
)

# Histograma de probabilidades de churn retornadas pelo modelo
PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribuição das probabilidades de churn preditas",
    buckets=[
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ],
)

# Contador de detecções de data drift por feature
DRIFT_DETECTIONS_TOTAL = Counter(
    "drift_detections_total",
    "Total de detecções de data drift por feature",
    ["feature", "drift_detected"],
)


def metrics_exposition() -> bytes:
    """Gera o conteúdo no formato de exposição do Prometheus.

    Returns:
        Bytes com métricas no Content-Type
        ``application/openmetrics-text`` (ou ``text/plain``).
    """
    return generate_latest()

"""Funções auxiliares para carga e exploração de métricas da API.

Este módulo encapsula a lógica de teste de carga e consulta ao
Prometheus Server, permitindo que scripts em `src/pipelines/`
permaneçam como simples drivers de execução.

Todas as métricas exibidas são consultadas ao Prometheus Server
via PromQL, garantindo que o relatório reflita os dados coletados
pela TSDB.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Any

import httpx

_API_URL = "http://localhost:8000"
_PROM_URL = "http://localhost:9090"


def healthcheck() -> bool:
    """Verifica se a API está respondendo em /health."""
    try:
        r = httpx.get(f"{_API_URL}/health", timeout=5)
        return r.status_code == 200  # noqa: PLR2004
    except httpx.RequestError:
        return False


def send_predict() -> None:
    """Envia uma requisição POST /predict com dados variados.

    Apenas dispara a requisição; métricas são coletadas pelo
    Prometheus via scraping do endpoint /metrics da API.
    """
    payload = {
        "customerID": f"CUST-{random.randint(1000, 9999)}",
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.randint(0, 1),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": random.randint(0, 72),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
        "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
        "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
        "DeviceProtection": random.choice(
            ["Yes", "No", "No internet service"]
        ),
        "TechSupport": random.choice(["Yes", "No", "No internet service"]),
        "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
        "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
        "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
        "MonthlyCharges": round(random.uniform(18.0, 120.0), 2),
    }
    httpx.post(
        f"{_API_URL}/predict", json=payload, timeout=10
    ).raise_for_status()


def run_load(num_requests: int) -> int:
    """Executa o load test e retorna quantidade de erros."""
    print(f"Iniciando carga: {num_requests} requisições...")
    errors = 0

    for i in range(num_requests):
        try:
            send_predict()
        except httpx.RequestError as e:
            errors += 1
            print(f"  [ERROR] req {i + 1}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_requests} concluídas...")

    return errors


def run_continuous_load(rate_per_sec: float) -> None:
    """Executa load test contínuo em loop infinito.

    Dispara requisições a uma taxa fixa e imprime um contador
    a cada segundo. Para com Ctrl+C.

    Args:
        rate_per_sec: Quantidade de requisições por segundo.
    """
    interval = 1.0 / rate_per_sec
    total = 0
    errors = 0
    start = time.perf_counter()
    next_report = start + 1.0

    print(
        f"[WATCH] Enviando ~{rate_per_sec} req/s contínuamente. "
        "Pressione Ctrl+C para parar."
    )

    try:
        while True:
            loop_start = time.perf_counter()

            try:
                send_predict()
                total += 1
            except httpx.RequestError:
                errors += 1

            # Report a cada segundo
            now = time.perf_counter()
            if now >= next_report:
                elapsed = int(now - start)
                print(
                    f"  [{elapsed:4d}s] Total: {total:5d} reqs | "
                    f"Erros: {errors:3d} | "
                    f"Taxa real: ~{total // max(elapsed, 1)} req/s",
                    end="\r",
                    file=sys.stderr,
                )
                next_report = now + 1.0

            # Sleep para manter a taxa
            sleep_time = interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        elapsed = max(int(time.perf_counter() - start), 1)
        print(
            f"\n[WATCH] Parado. Total: {total} reqs em {elapsed}s "
            f"(~{total // elapsed} req/s). Erros: {errors}."
        )


def promql_query(query: str) -> list[dict[str, Any]]:
    """Executa uma query PromQL no Prometheus Server."""
    try:
        r = httpx.get(
            f"{_PROM_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("data", {}).get("result", [])
    except httpx.RequestError:
        return []


def print_prometheus_report(errors: int) -> None:
    """Consulta o Prometheus e imprime métricas agregadas."""
    print("\n" + "=" * 50)
    print("RELATÓRIO DE MÉTRICAS - PROMETHEUS TSDB")
    print("=" * 50)

    # Requisições totais por endpoint
    results = promql_query(
        "sum by (path, method, status_code) (http_requests_total)"
    )
    if results:
        print("\nRequisições totais (Prometheus):")
        for res in results:
            labels = res["metric"]
            value = res["value"][1]
            print(
                f"  {labels.get('method', '?')} {labels.get('path', '?')} "
                f"[{labels.get('status_code', '?')}] -> {value}"
            )
    else:
        print("\n[AVISO] Prometheus não respondeu ou não há dados ainda.")
        print(
            "        Verifique se o container 'churn-prometheus' está no ar."
        )
        print(f"        Acesse {_PROM_URL}/targets para verificar.")
        return

    # Erros detectados localmente (requests que falharam antes de chegar à API)
    if errors:
        print(f"\nErros detectados localmente (timeout/connection): {errors}")

    # Latência média por endpoint
    # (calculada pelo Prometheus a partir dos histogramas)
    results = promql_query(
        "rate(http_request_duration_seconds_sum[5m]) "
        "/ rate(http_request_duration_seconds_count[5m])"
    )
    if results:
        print("\nLatência média últimos 5m (Prometheus):")
        for res in results:
            labels = res["metric"]
            value = float(res["value"][1]) * 1000
            print(
                f"  {labels.get('method', '?')} {labels.get('path', '?')} "
                f"-> {value:.2f} ms"
            )

    # Total de predições no histograma
    results = promql_query("prediction_probability_count")
    if results:
        print("\nTotal de predições no histograma:")
        for res in results:
            value = res["value"][1]
            print(f"  prediction_probability_count = {value}")

    # Distribuição de probabilidades (buckets do histograma)
    results = promql_query("prediction_probability_bucket")
    if results:
        print("\nDistribuição de probabilidades de churn:")
        for res in sorted(results, key=lambda x: float(x["metric"]["le"])):
            le = res["metric"]["le"]
            value = res["value"][1]
            print(f"  <= {le:4s} -> {value:6s} predições")

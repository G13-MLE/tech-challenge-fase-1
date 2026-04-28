"""Script de carga e exploração de métricas da API.

Modos de operação:
    Batch (padrão): envia N requisições e gera relatório via Prometheus.
    Watch (--watch): envia requisições continuamente em tempo real
                     para observação no Grafana.

Este módulo NÃO deve conter definições de funções — toda a lógica
reutilizável está em src.api.monitoring.load_tester.

Uso:
    # Modo batch (padrão): 100 reqs + relatório
    uv run python -m src.pipelines.explore_metrics

    # Modo watch contínuo: ~2 req/s para sempre
    uv run python -m src.pipelines.explore_metrics --watch --rate 2
"""

from __future__ import annotations

import argparse
import sys
import time

from src.api.monitoring.load_tester import (
    healthcheck,
    print_prometheus_report,
    run_continuous_load,
    run_load,
)

_DEFAULT_REQS = 100
_DEFAULT_RATE = 1.0

parser = argparse.ArgumentParser(
    description="Script de carga e exploração de métricas"
)
parser.add_argument(
    "--requests",
    type=int,
    default=_DEFAULT_REQS,
    help=f"Número de requisições no modo batch (padrão: {_DEFAULT_REQS})",
)
parser.add_argument(
    "--skip-load",
    action="store_true",
    help="Pula a fase de carga e só consulta Prometheus (modo batch)",
)
parser.add_argument(
    "--watch",
    action="store_true",
    help=(
        "Modo contínuo: envia requisições em loop infinito "
        "para observação em tempo real no Grafana"
    ),
)
parser.add_argument(
    "--rate",
    type=float,
    default=_DEFAULT_RATE,
    help=(f"Requisições por segundo no modo watch (padrão: {_DEFAULT_RATE})"),
)
args = parser.parse_args()

if not healthcheck():
    print(
        "[ERROR] API não responde em http://localhost:8000/health\n"
        "        Certifique-se de que a API está rodando:\n"
        "        uv run fastapi dev src/api/main.py\n"
        "        ou\n"
        "        docker compose -f docker/docker-compose.api.yml up"
    )
    sys.exit(1)

print("[OK] API respondendo em http://localhost:8000")

if args.watch:
    # Modo contínuo: fire-and-forget para observação no Grafana
    run_continuous_load(args.rate)

    # Após Ctrl+C, mostra relatório final do Prometheus
    print(
        "\n[AGUARDE] O Prometheus coleta métricas a cada 15s. "
        "Aguardando 20s para relatório final..."
    )
    time.sleep(20)
    print_prometheus_report(errors=0)

else:
    # Modo batch: envia N reqs e gera relatório
    if not args.skip_load:
        errors = run_load(args.requests)
        print(
            "\n[AGUARDE] O Prometheus coleta métricas a cada 15s. "
            "Aguardando 20s para garantir que os dados estejam na TSDB..."
        )
        time.sleep(20)
    else:
        errors = 0

    print_prometheus_report(errors)

print("\n" + "=" * 50)
print("DICA: Acesse o Grafana em http://localhost:3000")
print("      Dashboard: API Churn - Métricas Operacionais")
print("=" * 50)

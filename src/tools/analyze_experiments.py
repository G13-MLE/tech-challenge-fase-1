"""Ferramenta para analisar experimentos e runs do MLflow.

Este modulo permite inspecionar runs do MLflow, comparar metricas
entre experimentos e exportar resultados para CSV.

Exemplos de uso:
    # Modo resumo: lista todos os experimentos e runs
    uv run python -m src.tools.analyze_experiments

    # Modo detalhe: inspeciona uma run especifica
    uv run python -m src.tools.analyze_experiments --run-id abc123

    # Exportar para CSV
    uv run python -m src.tools.analyze_experiments
    # saida padrao: reports/mlflow_analysis.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Sequence

load_dotenv()

_DEFAULT_URI = "http://localhost:5000"
DEFAULT_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    _DEFAULT_URI,
)
EXPERIMENT_NAMES_ENV = [
    os.getenv(
        "MLFLOW_DUMMY_EXPERIMENT_NAME",
        "tech-challenge-dummy-baseline",
    ),
    os.getenv(
        "MLFLOW_MLP_EXPERIMENT_NAME",
        "tech-challenge-mlp",
    ),
    os.getenv(
        "MLFLOW_LOGISTIC_EXPERIMENT_NAME",
        "tech-challenge-logistic-regression",
    ),
]

_PRIORITY_METRICS = [
    "f1_score",
    "test_f1_score",
    "val_f1",
    "roc_auc",
    "test_roc_auc",
    "accuracy",
    "test_accuracy",
    "precision",
    "recall",
    "pr_auc",
    "test_pr_auc",
]

_MAX_DISPLAY_METRICS = 10

console = Console()


@dataclass(frozen=True)
class RunSummary:
    """Resumo de uma run do MLflow."""

    run_id: str
    run_name: str
    experiment_name: str
    status: str
    start_time: str
    metrics: dict[str, float]
    params: dict[str, str]


def format_timestamp(ts: object | None) -> str:
    """Formata um timestamp unix para string legivel."""
    if ts is None:
        return "N/A"
    ts_sec: float
    if hasattr(ts, "timestamp"):
        ts_sec = ts.timestamp()  # type: ignore[union-attr]
    else:
        ts_sec = float(cast(float, ts)) / 1000
    return datetime.fromtimestamp(ts_sec, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def get_experiment_names() -> list[str]:
    """Retorna a lista de nomes de experimentos configurados no .env."""
    names = [name for name in EXPERIMENT_NAMES_ENV if name]
    if not names:
        return ["tech-challenge-default"]
    return names


def _find_experiment_id_by_name(name: str) -> str | None:
    """Busca o ID de um experimento pelo nome."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return None
    return experiment.experiment_id


def _fetch_runs_for_experiment(exp_name: str) -> list[RunSummary]:
    """Busca todas as runs finalizadas de um experimento."""
    exp_id = _find_experiment_id_by_name(exp_name)
    if exp_id is None:
        return []

    runs_df = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string="status = 'FINISHED'",
        ),
    )

    if runs_df.empty:
        return []

    summaries: list[RunSummary] = []
    for _, row_raw in runs_df.iterrows():
        raw_dict = cast("pd.Series", row_raw).to_dict()
        row = cast(dict[str, object], raw_dict)

        def _filter_prefix(
            row_dict: dict[str, object], prefix: str
        ) -> dict[str, object]:
            return {
                k: v
                for k, v in row_dict.items()
                if isinstance(k, str)
                and k.startswith(prefix)
                and v is not None
            }

        metrics = _filter_prefix(row, "metrics.")
        params = _filter_prefix(row, "params.")

        summaries.append(
            RunSummary(
                run_id=str(row.get("run_id", "")),
                run_name=str(row.get("tags.mlflow.runName", "N/A")),
                experiment_name=exp_name,
                status=str(row.get("status", "N/A")),
                start_time=format_timestamp(
                    cast(float | int | None, row.get("start_time"))
                ),
                metrics={
                    str(k).replace("metrics.", ""): cast(float, v)
                    for k, v in metrics.items()
                },
                params={
                    str(k).replace("params.", ""): str(v)
                    for k, v in params.items()
                },
            )
        )

    return summaries


def _print_experiment_summary(runs: list[RunSummary]) -> None:
    """Imprime uma tabela resumo de runs no terminal."""
    if not runs:
        console.print("[WARN] Nenhuma run encontrada para este experimento.")
        return

    # Coleta todas as metricas unicas para colunas
    all_metrics: set[str] = set()
    for run in runs:
        all_metrics.update(run.metrics.keys())

    # Exibe apenas metricas prioritarias (evita overflow no terminal)
    sorted_metrics = [m for m in _PRIORITY_METRICS if m in all_metrics]
    # Fallback: se nenhuma prioritaria existir, mostra metricas alfabeticas
    if not sorted_metrics:
        sorted_metrics = sorted(all_metrics)[:_MAX_DISPLAY_METRICS]
    # Segundo fallback: se ainda tiver muitas, trunca
    elif len(sorted_metrics) > _MAX_DISPLAY_METRICS:
        sorted_metrics = sorted_metrics[:_MAX_DISPLAY_METRICS]

    table = Table(title=f"Experimento: {runs[0].experiment_name}")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Nome", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Inicio", style="yellow")
    for metric in sorted_metrics:
        table.add_column(metric, justify="right")

    # Identifica a melhor run por f1_score
    best_run_id: str | None = None
    if "f1_score" in sorted_metrics:
        best = max(runs, key=lambda r: r.metrics.get("f1_score", -1))
        best_run_id = best.run_id
    elif "test_f1_score" in sorted_metrics:
        best = max(runs, key=lambda r: r.metrics.get("test_f1_score", -1))
        best_run_id = best.run_id

    for run in runs:
        row = [
            run.run_id[:8],
            run.run_name,
            run.status,
            run.start_time,
        ]
        for metric in sorted_metrics:
            val = run.metrics.get(metric)
            cell = f"{val:.4f}" if val is not None else "N/A"
            if run.run_id == best_run_id:
                cell = f"[bold green]{cell}[/bold green]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)

    if best_run_id:
        best_f1 = max(
            r.metrics.get("f1_score", r.metrics.get("test_f1_score", -1.0))
            for r in runs
        )
        console.print(
            f"[OK] Melhor run: {best_run_id[:8]} (f1_score={best_f1:.4f})"
        )


def _get_run_details(run_id: str) -> dict[str, object] | None:
    """Obtem todos os detalhes de uma run especifica."""
    try:
        run = mlflow.get_run(run_id)
    except MlflowException:
        return None

    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": format_timestamp(run.info.start_time),
        "end_time": format_timestamp(run.info.end_time),
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def _print_run_details(details: dict[str, object]) -> None:
    """Imprime os detalhes completos de uma run."""
    console.print(f"\n[bold cyan]Run: {details['run_name']}[/bold cyan]")
    console.print(f"ID: {details['run_id']}")
    console.print(f"Status: {details['status']}")
    console.print(f"Inicio: {details['start_time']}")
    console.print(f"Fim: {details['end_time']}")

    metrics = cast(dict[str, float], details["metrics"])
    if metrics:
        table = Table(title="Metricas")
        table.add_column("Metrica", style="cyan")
        table.add_column("Valor", justify="right")
        for k, v in sorted(metrics.items()):
            table.add_row(k, f"{v:.6f}")
        console.print(table)

    params = cast(dict[str, str], details["params"])
    if params:
        table = Table(title="Parametros")
        table.add_column("Parametro", style="magenta")
        table.add_column("Valor")
        for k, v in sorted(params.items()):  # type: ignore[assignment]
            table.add_row(k, str(v))
        console.print(table)


def _build_dataframe(runs: list[RunSummary]) -> pd.DataFrame:
    """Constroi um DataFrame consolidado de runs para exportacao CSV."""
    rows: list[dict[str, str | float]] = []
    for run in runs:
        row: dict[str, str | float] = {
            "experiment_name": run.experiment_name,
            "run_id": run.run_id,
            "run_name": run.run_name,
            "status": run.status,
            "start_time": run.start_time,
        }
        for k, v in run.params.items():  # type: ignore[assignment]
            row[f"param.{k}"] = v
        for k, v in run.metrics.items():  # type: ignore[assignment]
            row[f"metric.{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_experiments(
    run_id: str | None = None,
    output_csv: str | None = None,
) -> int:
    """Executa a analise dos experimentos do MLflow.

    Args:
        run_id: ID especifico de uma run para detalhamento.
        output_csv: Caminho para salvar o CSV de saida.

    Returns:
        Codigo de saida (0 = sucesso, 1 = erro).
    """
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    if run_id:
        details = _get_run_details(run_id)
        if details is None:
            console.print(f"[ERROR] Run nao encontrada: {run_id}")
            return 1
        _print_run_details(details)
        return 0

    experiment_names = get_experiment_names()
    all_runs: list[RunSummary] = []

    for exp_name in experiment_names:
        runs = _fetch_runs_for_experiment(exp_name)
        if runs:
            _print_experiment_summary(runs)
            all_runs.extend(runs)
        else:
            console.print(f"[WARN] Nenhuma run encontrada para: {exp_name}")

    if output_csv and all_runs:
        df = _build_dataframe(all_runs)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        console.print(f"[OK] CSV salvo em: {output_csv}")

    return 0


def parse_args(
    args: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisar experimentos e runs do MLflow"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID de uma run especifica para detalhamento",
    )
    _DEFAULT_OUTPUT = "reports/mlflow_analysis.csv"
    parser.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_OUTPUT,
        help=(f"Caminho para salvar o CSV (padrao: {_DEFAULT_OUTPUT})"),
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """Ponto de entrada principal da ferramenta."""
    parsed = parse_args(args)
    return analyze_experiments(
        run_id=parsed.run_id,
        output_csv=parsed.output,
    )


if __name__ == "__main__":
    sys.exit(main())

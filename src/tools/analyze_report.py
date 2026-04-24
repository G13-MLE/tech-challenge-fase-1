"""Automatiza a analise comparativa de experimentos ML a partir do CSV.

Le `reports/mlflow_analysis.csv` e gera um relatorio estruturado
comparando Dummy Baseline e MLP, avaliando metricas, overfitting
e underfitting.

Uso:
    uv run python -m src.tools.analyze_report
    uv run python -m src.tools.analyze_report --output reports/comparacao.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Mapping

console = Console()

_DEFAULT_CSV = "reports/mlflow_analysis.csv"
_DEFAULT_OUTPUT = "reports/experiment_comparison.md"

_METRIC_COLS = {
    "accuracy": "metric.accuracy",
    "f1_score": "metric.f1_score",
    "precision": "metric.precision",
    "recall": "metric.recall",
    "roc_auc": "metric.roc_auc",
    "pr_auc": "metric.pr_auc",
    "brier_score": "metric.brier_score",
}

_TEST_METRIC_COLS = {
    "test_accuracy": "metric.test_accuracy",
    "test_f1_score": "metric.test_f1_score",
    "test_precision": "metric.test_precision",
    "test_recall": "metric.test_recall",
    "test_roc_auc": "metric.test_roc_auc",
    "test_pr_auc": "metric.test_pr_auc",
    "test_brier_score": "metric.test_brier_score",
}

_VAL_METRIC_COLS = {
    "val_loss": "metric.val_loss",
    "val_f1": "metric.val_f1",
    "val_auc": "metric.val_auc",
}

# --- Constantes para diagnosticos de overfitting ---
_GAP_AUC_LOW = 0.01
_GAP_LOSS_LOW = 0.05
_GAP_AUC_HIGH = 0.02
_GAP_LOSS_HIGH = 0.1
_MIN_SAMPLES_STD = 2


def _fmt(v: float | None) -> str:
    """Formata valor numerico ou retorna N/A."""
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:.4f}"


def _safe_mean(series: pd.Series) -> float | None:
    """Media segura ignorando NaNs."""
    s = series.dropna()
    if s.empty:
        return None
    return float(s.mean())  # type: ignore[return-value]


def _safe_std(series: pd.Series) -> float | None:
    """Desvio padrao seguro."""
    s = series.dropna()
    if s.empty or len(s) < _MIN_SAMPLES_STD:
        return None
    return float(s.std(ddof=0))  # type: ignore[return-value]


def load_data(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV de analise MLflow."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Arquivo CSV vazio: {csv_path}")
    return df


def split_experiments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa DataFrame em Dummy e MLP."""
    dummy_mask = df["experiment_name"].str.contains(
        "dummy", case=False, na=False
    )
    mlp_mask = df["experiment_name"].str.contains("mlp", case=False, na=False)
    dummy_df = df[dummy_mask].copy()
    mlp_df = df[mlp_mask].copy()
    return dummy_df, mlp_df


def analyze_dummy(dummy_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Analisa runs dummy agrupadas por estrategia."""
    strategy_col = "param.strategy"
    if strategy_col not in dummy_df.columns:
        return {}

    results: dict[str, pd.DataFrame] = {}
    for strategy, group in dummy_df.groupby(strategy_col):
        stats: dict[str, list[Any]] = {
            "metric": [],
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
        }
        for name, col in _METRIC_COLS.items():
            if col not in group.columns:
                continue
            s = pd.to_numeric(group[col], errors="coerce")
            stats["metric"].append(name)
            stats["mean"].append(_safe_mean(s))
            stats["std"].append(_safe_std(s))
            stats["min"].append(
                float(s.min()) if not s.dropna().empty else None  # type: ignore[arg-type]
            )
            stats["max"].append(
                float(s.max()) if not s.dropna().empty else None  # type: ignore[arg-type]
            )
        results[str(strategy)] = pd.DataFrame(stats)
    return results


def analyze_mlp(mlp_df: pd.DataFrame) -> dict[str, Any]:
    """Analisa runs MLP retornando estatisticas e flags."""
    test_auc_col = _TEST_METRIC_COLS["test_roc_auc"]
    valid_mlp = mlp_df[mlp_df[test_auc_col].notna()].copy()

    stats: dict[str, pd.DataFrame] = {}

    # --- Metricas de teste ---
    test_stats: dict[str, list[Any]] = {
        "metric": [],
        "mean": [],
        "std": [],
        "min": [],
        "median": [],
        "max": [],
    }
    for name, col in _TEST_METRIC_COLS.items():
        if col not in valid_mlp.columns:
            continue
        s = pd.to_numeric(valid_mlp[col], errors="coerce")
        test_stats["metric"].append(name)
        test_stats["mean"].append(_safe_mean(s))
        test_stats["std"].append(_safe_std(s))
        test_stats["min"].append(
            float(s.min()) if not s.dropna().empty else None  # type: ignore[arg-type]
        )
        test_stats["median"].append(
            float(s.median()) if not s.dropna().empty else None  # type: ignore[arg-type]
        )
        test_stats["max"].append(
            float(s.max()) if not s.dropna().empty else None  # type: ignore[arg-type]
        )
    stats["test"] = pd.DataFrame(test_stats)

    # --- Gaps de overfitting ---
    train_loss = pd.to_numeric(valid_mlp["metric.train_loss"], errors="coerce")
    val_loss = pd.to_numeric(valid_mlp["metric.val_loss"], errors="coerce")
    val_auc = pd.to_numeric(valid_mlp["metric.val_auc"], errors="coerce")
    test_auc = pd.to_numeric(valid_mlp[test_auc_col], errors="coerce")

    gap_loss = (val_loss - train_loss).dropna()
    gap_auc = (val_auc - test_auc).dropna()

    overfitting = {
        "mean_train_loss": train_loss.mean(),
        "mean_val_loss": val_loss.mean(),
        "gap_loss_mean": gap_loss.mean() if not gap_loss.empty else None,
        "gap_loss_std": gap_loss.std(ddof=0) if not gap_loss.empty else None,
        "mean_val_auc": val_auc.mean(),
        "mean_test_auc": test_auc.mean(),
        "gap_auc_mean": gap_auc.mean() if not gap_auc.empty else None,
        "gap_auc_std": gap_auc.std(ddof=0) if not gap_auc.empty else None,
    }

    # --- Melhor run por ROC-AUC ---
    best_idx = test_auc.idxmax() if not test_auc.empty else None
    best_run = valid_mlp.loc[best_idx] if best_idx is not None else None

    return {
        "test_stats": stats["test"],
        "overfitting": overfitting,
        "best_run": best_run,
        "num_runs": len(valid_mlp),
    }


def _render_dummy_section(
    dummy_stats: Mapping[str, pd.DataFrame],
) -> tuple[list[str], list[str]]:
    """Retorna linhas Markdown e tabelas Rich para Dummy."""
    lines: list[str] = []
    md_rows: list[str] = []

    sec_dummy = "## 1. Dummy Baseline"
    console.print(f"[bold]{sec_dummy}[/bold]")
    lines.append(sec_dummy)
    lines.append("")

    for strategy, df_stat in dummy_stats.items():
        console.print(f"\n[bold magenta]Estrategia: {strategy}[/bold magenta]")
        lines.append(f"### Estrategia: {strategy}")
        lines.append("")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Metrica")
        table.add_column("Media", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        md_rows = [
            "| Metrica | Media | Std | Min | Max |",
            "|---------|-------|-----|-----|-----|",
        ]

        for _, row in df_stat.iterrows():
            table.add_row(
                str(row["metric"]),
                _fmt(row["mean"]),
                _fmt(row["std"]),
                _fmt(row["min"]),
                _fmt(row["max"]),
            )
            md_rows.append(
                f"| {row['metric']} | {_fmt(row['mean'])} | "
                f"{_fmt(row['std'])} | {_fmt(row['min'])} | "
                f"{_fmt(row['max'])} |"
            )

        console.print(table)
        lines.extend(md_rows)
        lines.append("")

    return lines, md_rows


def _render_mlp_section(
    mlp_stats: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Retorna linhas Markdown e tabelas Rich para MLP."""
    lines: list[str] = []
    md_rows: list[str] = []

    sec_mlp = "## 2. MLP"
    console.print(f"\n[bold]{sec_mlp}[/bold]")
    lines.append(sec_mlp)
    lines.append("")

    num_runs = mlp_stats.get("num_runs", 0)
    console.print(f"Runs validos analisados: [bold]{num_runs}[/bold]")
    lines.append(f"**Runs validos analisados:** {num_runs}")
    lines.append("")

    test_stats: pd.DataFrame = mlp_stats["test_stats"]  # type: ignore[assignment]
    console.print("\n[bold]Metricas de Teste (MLP)[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metrica")
    table.add_column("Media", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Mediana", justify="right")
    table.add_column("Max", justify="right")

    md_rows = [
        "| Metrica | Media | Std | Min | Mediana | Max |",
        "|---------|-------|-----|-----|---------|-----|",
    ]

    for _, row in test_stats.iterrows():
        table.add_row(
            str(row["metric"]),
            _fmt(row["mean"]),
            _fmt(row["std"]),
            _fmt(row["min"]),
            _fmt(row["median"]),
            _fmt(row["max"]),
        )
        md_rows.append(
            f"| {row['metric']} | {_fmt(row['mean'])} | "
            f"{_fmt(row['std'])} | {_fmt(row['min'])} | "
            f"{_fmt(row['median'])} | {_fmt(row['max'])} |"
        )

    console.print(table)
    lines.extend(md_rows)
    lines.append("")

    return lines, md_rows


def _render_overfitting_section(
    mlp_stats: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Retorna linhas Markdown e tabelas Rich para overfitting."""
    lines: list[str] = []
    md_rows: list[str] = []

    sec_of = "## 3. Analise de Overfitting / Underfitting"
    console.print(f"\n[bold]{sec_of}[/bold]")
    lines.append(sec_of)
    lines.append("")

    of: dict[str, float | None] = mlp_stats["overfitting"]  # type: ignore[assignment]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Indicador")
    table.add_column("Valor", justify="right")

    md_rows = [
        "| Indicador | Valor |",
        "|-----------|-------|",
    ]

    items = [
        ("Train Loss (medio)", "mean_train_loss"),
        ("Val Loss (medio)", "mean_val_loss"),
        ("Gap Loss medio (val - train)", "gap_loss_mean"),
        ("Gap Loss std", "gap_loss_std"),
        ("Val AUC (medio)", "mean_val_auc"),
        ("Test AUC (medio)", "mean_test_auc"),
        ("Gap AUC medio (val - test)", "gap_auc_mean"),
        ("Gap AUC std", "gap_auc_std"),
    ]

    for label, key in items:
        val = of.get(key)
        cell = _fmt(val)  # type: ignore[arg-type]
        table.add_row(label, cell)
        md_rows.append(f"| {label} | {cell} |")

    console.print(table)
    lines.extend(md_rows)
    lines.append("")

    gap_auc = of.get("gap_auc_mean")
    gap_loss = of.get("gap_loss_mean")

    if gap_auc is not None and gap_loss is not None:
        if gap_auc < _GAP_AUC_LOW and gap_loss < _GAP_LOSS_LOW:
            diag = (
                "[OK] Overfitting MINIMO: gaps treino/validacao e "
                "validacao/teste sao pequenos."
            )
        elif gap_auc > _GAP_AUC_HIGH or gap_loss > _GAP_LOSS_HIGH:
            diag = (
                "[WARN] Possivel overfitting detectado: "
                "gap entre validacao e teste elevado."
            )
        else:
            diag = "[INFO] Overfitting leve, dentro de limites aceitaveis."
    else:
        diag = "[WARN] Dados insuficientes para diagnosticar overfitting."

    console.print(f"\n[bold]{diag}[/bold]\n")
    stripped = (
        diag.replace("[OK]", "").replace("[WARN]", "").replace("[INFO]", "")
    )
    lines.append(f"**Diagnostico:** {stripped}")
    lines.append("")

    return lines, md_rows


def _render_best_run_section(
    mlp_stats: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Retorna linhas Markdown e tabelas Rich para melhor run."""
    lines: list[str] = []
    md_rows: list[str] = []

    best_run = mlp_stats.get("best_run")
    if best_run is not None and not isinstance(best_run, float):
        sec_best = "## 4. Melhor Run MLP (por Test ROC-AUC)"
        console.print(f"[bold]{sec_best}[/bold]")
        lines.append(sec_best)
        lines.append("")

        run_name = str(best_run.get("run_name", "N/A"))
        run_id_short = str(best_run.get("run_id", "N/A"))[:8]
        test_auc_val = best_run.get(_TEST_METRIC_COLS["test_roc_auc"], "N/A")
        test_f1_val = best_run.get(_TEST_METRIC_COLS["test_f1_score"], "N/A")

        console.print(f"Run : {run_name} ({run_id_short})")
        console.print(f"Test ROC-AUC: {test_auc_val}")
        console.print(f"Test F1-Score: {test_f1_val}")

        lines.append(f"- **Run:** {run_name} (`{run_id_short}`)")
        lines.append(f"- **Test ROC-AUC:** {test_auc_val}")
        lines.append(f"- **Test F1-Score:** {test_f1_val}")
        lines.append("")

    return lines, md_rows


def print_report(
    dummy_stats: Mapping[str, pd.DataFrame],
    mlp_stats: dict[str, Any],
) -> str:
    """Imprime relatorio no terminal e retorna texto Markdown."""
    lines: list[str] = []

    header = "# Relatorio de Analise: Dummy Baseline vs MLP"
    console.print(f"\n[bold cyan]{header}[/bold cyan]\n")
    lines.append(header)
    lines.append("")

    # === DUMMY ===
    _ = _render_dummy_section(dummy_stats)
    lines.extend(_[0])

    # === MLP ===
    _ = _render_mlp_section(mlp_stats)
    lines.extend(_[0])

    # === Overfitting ===
    _ = _render_overfitting_section(mlp_stats)
    lines.extend(_[0])

    # === Melhor run ===
    _ = _render_best_run_section(mlp_stats)
    lines.extend(_[0])

    return "\n".join(lines)


def main(args: Sequence[str] | None = None) -> int:
    """Ponto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Analisa relatorio MLflow CSV e compara experimentos"
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_CSV,
        help=f"Caminho do CSV MLflow (padrao: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=(
            f"Caminho para salvar relatorio Markdown "
            f"(padrao: {_DEFAULT_OUTPUT})"
        ),
    )
    parsed = parser.parse_args(args)

    csv_path = Path(parsed.input)
    if not csv_path.exists():
        console.print(f"[ERROR] Arquivo nao encontrado: {csv_path}")
        return 1

    df = load_data(str(csv_path))
    dummy_df, mlp_df = split_experiments(df)

    if dummy_df.empty:
        console.print("[WARN] Nenhuma run Dummy encontrada.")
    if mlp_df.empty:
        console.print("[WARN] Nenhuma run MLP encontrada.")

    dummy_stats = analyze_dummy(dummy_df)
    mlp_stats = analyze_mlp(mlp_df)

    markdown = print_report(dummy_stats, mlp_stats)

    if parsed.output:
        out_path = Path(parsed.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        console.print(f"[OK] Relatorio salvo em: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Testes para a ferramenta de analise de experimentos MLflow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.tools.analyze_experiments import (
    analyze_experiments,
    format_timestamp,
    parse_args,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_mlflow() -> Generator[MagicMock, None, None]:
    """Fixture que munda a URI do tracking para evitar conexoes reais."""
    with patch("src.tools.analyze_experiments.mlflow") as m:
        m.set_tracking_uri = MagicMock()
        yield m


def test_parse_args_default() -> None:
    """Testa o parsing de argumentos padrao."""
    args = parse_args([])
    assert args.run_id is None
    assert args.output == "reports/mlflow_analysis.csv"


def test_parse_args_with_run_id() -> None:
    """Testa o parsing com run-id especificado."""
    args = parse_args(["--run-id", "abc123"])
    assert args.run_id == "abc123"


def test_analyze_experiments_summary(
    mock_mlflow: MagicMock, tmp_path: Path
) -> None:
    """Testa o modo resumo que lista experimentos e runs."""
    exp_mock = MagicMock()
    exp_mock.experiment_id = "1"
    mock_mlflow.get_experiment_by_name.return_value = exp_mock

    ts = int(datetime.now(tz=UTC).timestamp() * 1000)
    runs_df = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "status": ["FINISHED"],
            "start_time": [ts],
            "metrics.f1_score": [0.85],
            "metrics.accuracy": [0.90],
            "params.lr": ["0.01"],
            "tags.mlflow.runName": ["test-run"],
        }
    )
    mock_mlflow.search_runs.return_value = runs_df

    output = tmp_path / "test.csv"
    # Isola a lista de experimentos para 1 so
    with patch(
        "src.tools.analyze_experiments.get_experiment_names",
        return_value=["tech-challenge-dummy-baseline"],
    ):
        code = analyze_experiments(run_id=None, output_csv=str(output))
    assert code == 0
    assert output.exists()

    df = pd.read_csv(str(output))
    assert len(df) == 1
    assert df.iloc[0]["experiment_name"] == "tech-challenge-dummy-baseline"
    assert df.iloc[0]["metric.f1_score"] == pytest.approx(0.85)


def test_analyze_experiments_empty_experiment(
    mock_mlflow: MagicMock,
) -> None:
    """Testa quando um experimento nao possui runs."""
    exp_mock = MagicMock()
    exp_mock.experiment_id = "1"
    mock_mlflow.get_experiment_by_name.return_value = exp_mock
    mock_mlflow.search_runs.return_value = pd.DataFrame()

    code = analyze_experiments(run_id=None, output_csv=None)
    assert code == 0


def test_analyze_experiments_run_details(
    mock_mlflow: MagicMock,
) -> None:
    """Testa o modo detalhe para uma run especifica."""
    run_mock = MagicMock()
    run_mock.info.run_id = "abc123"
    run_mock.info.run_name = "detailed-run"
    run_mock.info.status = "FINISHED"
    run_mock.info.start_time = int(datetime.now(tz=UTC).timestamp() * 1000)
    run_mock.info.end_time = run_mock.info.start_time
    run_mock.data.metrics = {"f1_score": 0.80}
    run_mock.data.params = {"lr": "0.01"}
    run_mock.data.tags = {"model": "mlp"}

    mock_mlflow.get_run.return_value = run_mock

    code = analyze_experiments(run_id="abc123", output_csv=None)
    assert code == 0
    mock_mlflow.get_run.assert_called_once_with("abc123")


def test_analyze_experiments_no_runs_all_experiments(
    mock_mlflow: MagicMock,
) -> None:
    """Testa quando todos os experimentos nao possuem runs."""
    mock_mlflow.get_experiment_by_name.return_value = None

    code = analyze_experiments(run_id=None, output_csv=None)
    assert code == 0


def test_analyze_experiments_run_not_found(
    mock_mlflow: MagicMock,
) -> None:
    """Testa quando a run especificada nao existe."""
    mock_mlflow.get_run.side_effect = MlflowException("Run not found")

    code = analyze_experiments(run_id="invalid", output_csv=None)
    assert code == 1


def test_format_timestamp_none() -> None:
    """Testa formatacao de timestamp nulo."""
    assert format_timestamp(None) == "N/A"


def test_format_timestamp_with_pandas_timestamp() -> None:
    """Testa formatacao com pd.Timestamp."""
    ts = pd.Timestamp("2024-01-15 10:30:00", tz=UTC)
    result = format_timestamp(ts)
    assert "2024-01-15" in result
    assert "10:30:00" in result
    """Testa formatacao de timestamp valido."""
    ts = int(
        datetime(
            2024,
            1,
            15,
            10,
            30,
            0,
            tzinfo=UTC,
        ).timestamp()
        * 1000
    )
    result = format_timestamp(ts)
    assert "2024-01-15" in result
    assert "10:30:00" in result

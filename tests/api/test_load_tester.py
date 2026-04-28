"""Testes para src.api.monitoring.load_tester.

Mocka requisições HTTP para testar sem depender da API ou
Prometheus estarem no ar.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.api.monitoring.load_tester import (
    healthcheck,
    print_prometheus_report,
    promql_query,
    run_continuous_load,
    run_load,
    send_predict,
)

# ------------------------------------------------------------------
# healthcheck
# ------------------------------------------------------------------


def test_healthcheck_success() -> None:
    """healthcheck retorna True quando /health responde 200."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch(
        "src.api.monitoring.load_tester.httpx.get", return_value=mock_response
    ):
        assert healthcheck() is True


def test_healthcheck_failure_on_500() -> None:
    """healthcheck retorna False quando /health responde 500."""
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch(
        "src.api.monitoring.load_tester.httpx.get", return_value=mock_response
    ):
        assert healthcheck() is False


def test_healthcheck_failure_on_request_error() -> None:
    """healthcheck retorna False quando a conexão falha."""
    with patch(
        "src.api.monitoring.load_tester.httpx.get",
        side_effect=httpx.RequestError("Connection refused"),
    ):
        assert healthcheck() is False


# ------------------------------------------------------------------
# send_predict
# ------------------------------------------------------------------


def test_send_predict_success() -> None:
    """send_predict dispara POST /predict e não levanta exceção."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch(
        "src.api.monitoring.load_tester.httpx.post", return_value=mock_response
    ) as mock_post:
        send_predict()
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0].endswith("/predict")
        assert "json" in kwargs
        assert "customerID" in kwargs["json"]


def test_send_predict_failure() -> None:
    """send_predict propaga exceção quando POST falha."""
    with (
        patch(
            "src.api.monitoring.load_tester.httpx.post",
            side_effect=httpx.RequestError("Timeout"),
        ),
        pytest.raises(httpx.RequestError),
    ):
        send_predict()


# ------------------------------------------------------------------
# run_load
# ------------------------------------------------------------------


def test_run_load_zero_errors() -> None:
    """run_load retorna 0 erros quando todas as requisições sucedem."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch(
        "src.api.monitoring.load_tester.httpx.post", return_value=mock_response
    ):
        errors = run_load(5)
        assert errors == 0


def test_run_load_counts_errors() -> None:
    """run_load conta erros quando requisições falham."""
    with patch(
        "src.api.monitoring.load_tester.httpx.post",
        side_effect=httpx.RequestError("Timeout"),
    ):
        errors = run_load(3)
        assert errors == 3  # noqa: PLR2004


# ------------------------------------------------------------------
# promql_query
# ------------------------------------------------------------------


def test_promql_query_success() -> None:
    """promql_query parseia corretamente resposta do Prometheus."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "result": [
                {
                    "metric": {"path": "/predict", "method": "POST"},
                    "value": [1234567890, "42"],
                }
            ]
        }
    }

    with patch(
        "src.api.monitoring.load_tester.httpx.get", return_value=mock_response
    ):
        result = promql_query("up")
        assert len(result) == 1
        assert result[0]["metric"]["path"] == "/predict"
        assert result[0]["value"][1] == "42"


def test_promql_query_failure() -> None:
    """promql_query retorna lista vazia quando Prometheus não responde."""
    with patch(
        "src.api.monitoring.load_tester.httpx.get",
        side_effect=httpx.RequestError("Connection refused"),
    ):
        result = promql_query("up")
        assert result == []


# ------------------------------------------------------------------
# print_prometheus_report
# ------------------------------------------------------------------


def test_print_prometheus_report_with_data(capsys: Any) -> None:
    """print_prometheus_report imprime métricas quando há dados."""
    mock_result = [
        {
            "metric": {
                "path": "/predict",
                "method": "POST",
                "status_code": "200",
            },
            "value": [1234567890, "10"],
        }
    ]

    with patch(
        "src.api.monitoring.load_tester.promql_query",
        side_effect=[
            mock_result,  # http_requests_total
            [],  # latência (sem dados para simplificar)
            [
                {"metric": {}, "value": [0, "10"]}
            ],  # prediction_probability_count
            [],  # prediction_probability_bucket
        ],
    ):
        print_prometheus_report(errors=0)

    captured = capsys.readouterr()
    assert "RELATÓRIO DE MÉTRICAS" in captured.out
    assert "POST /predict [200] -> 10" in captured.out
    assert "prediction_probability_count = 10" in captured.out


def test_print_prometheus_report_without_data(capsys: Any) -> None:
    """print_prometheus_report avisa quando Prometheus não tem dados."""
    with patch("src.api.monitoring.load_tester.promql_query", return_value=[]):
        print_prometheus_report(errors=0)

    captured = capsys.readouterr()
    assert "[AVISO] Prometheus não respondeu" in captured.out


def test_print_prometheus_report_shows_local_errors(capsys: Any) -> None:
    """print_prometheus_report exibe erros detectados localmente."""
    mock_result = [
        {
            "metric": {
                "path": "/predict",
                "method": "POST",
                "status_code": "200",
            },
            "value": [1234567890, "5"],
        }
    ]

    with patch(
        "src.api.monitoring.load_tester.promql_query",
        side_effect=[
            mock_result,
            [],
            [{"metric": {}, "value": [0, "5"]}],
            [],
        ],
    ):
        print_prometheus_report(errors=3)

    captured = capsys.readouterr()
    assert "Erros detectados localmente" in captured.out
    assert "3" in captured.out


def test_print_prometheus_report_with_latency_and_buckets(capsys: Any) -> None:
    """print_prometheus_report imprime latência e distribuição de probs."""
    mock_requests = [
        {
            "metric": {
                "path": "/predict",
                "method": "POST",
                "status_code": "200",
            },
            "value": [0, "8"],
        }
    ]
    mock_latency = [
        {
            "metric": {"path": "/predict", "method": "POST"},
            "value": [0, "0.012"],
        }
    ]
    mock_count = [{"metric": {}, "value": [0, "8"]}]
    mock_buckets = [
        {"metric": {"le": "0.5"}, "value": [0, "5"]},
        {"metric": {"le": "1.0"}, "value": [0, "8"]},
    ]

    with patch(
        "src.api.monitoring.load_tester.promql_query",
        side_effect=[mock_requests, mock_latency, mock_count, mock_buckets],
    ):
        print_prometheus_report(errors=0)

    captured = capsys.readouterr()
    assert "Latência média" in captured.out
    assert "12.00" in captured.out  # 0.012s * 1000 = 12ms
    assert "prediction_probability_count = 8" in captured.out
    assert "Distribuição de probabilidades" in captured.out
    assert "<= 0.5" in captured.out


# ------------------------------------------------------------------
# run_continuous_load
# ------------------------------------------------------------------


def test_run_continuous_load_stops_on_keyboard_interrupt(capsys: Any) -> None:
    """run_continuous_load para graciosamente com Ctrl+C."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:  # noqa: PLR2004
            raise KeyboardInterrupt
        return mock_response

    with patch(
        "src.api.monitoring.load_tester.httpx.post", side_effect=side_effect
    ):
        run_continuous_load(rate_per_sec=100.0)

    captured = capsys.readouterr()
    assert "[WATCH] Parado" in captured.out
    assert "Total:" in captured.out


# ------------------------------------------------------------------
# run_load progresso
# ------------------------------------------------------------------


def test_run_load_prints_progress_every_20(capsys: Any) -> None:
    """run_load imprime progresso a cada 20 requisições."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch(
        "src.api.monitoring.load_tester.httpx.post", return_value=mock_response
    ):
        errors = run_load(25)

    assert errors == 0
    captured = capsys.readouterr()
    assert "20/25 concluídas" in captured.out

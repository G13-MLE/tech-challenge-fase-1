from __future__ import annotations

import pandas as pd

from src.tools.analyze_report import (
    analyze_dummy,
    analyze_logistic,
    analyze_mlp,
    load_data,
    print_report,
    split_experiments,
)

# -- Constantes para testes --
_N_ROWS = 3
_N_DUMMY = 1
_N_MLP = 1
_N_LOGISTIC = 1
_N_MLP_RUNS = 2
_N_LOGISTIC_RUNS = 2
_EXPECTED_DUMMY_ACCURACY = 0.50
_EXPECTED_AUC = 0.84
_EXPECTED_LOGISTIC_AUC = 0.82

# -- CSV com dados dos 3 experimentos --
_FULL_CSV = (
    "experiment_name,run_id,run_name,param.strategy,"
    "metric.accuracy,metric.roc_auc,"
    "metric.test_accuracy,metric.test_roc_auc,metric.test_f1_score,"
    "metric.val_auc,metric.train_loss,metric.val_loss,"
    "metric.cv_accuracy_mean,metric.cv_f1_mean,"
    "metric.test_precision,metric.test_recall\n"
    "tech-challenge-dummy-baseline,1,r1,uniform,0.5,0.5,,,,,,,,,\n"
    "tech-challenge-mlp,3,r3,,,,0.8,0.85,0.6,0.86,0.4,0.42,,,\n"
    "tech-challenge-logistic-regression,4,r4,,,,0.79,0.82,0.58,,"
    "0.80,0.62,0.61,0.55\n"
)


def test_load_and_split(tmp_path):
    csv_path = tmp_path / "mlflow_analysis.csv"
    csv_path.write_text(_FULL_CSV)
    df = load_data(str(csv_path))
    assert len(df) == _N_ROWS
    dummy_df, mlp_df, logistic_df = split_experiments(df)
    assert len(dummy_df) == _N_DUMMY
    assert len(mlp_df) == _N_MLP
    assert len(logistic_df) == _N_LOGISTIC


def test_analyze_dummy(tmp_path):
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text(
        "experiment_name,param.strategy,"
        "metric.accuracy,metric.roc_auc\n"
        "tech-challenge-dummy-baseline,uniform,0.48,0.5\n"
        "tech-challenge-dummy-baseline,uniform,0.52,0.5\n"
        "tech-challenge-dummy-baseline,most_frequent,0.73,0.5\n"
    )
    df = load_data(str(csv_path))
    dummy_df, _, _ = split_experiments(df)
    stats = analyze_dummy(dummy_df)
    assert "uniform" in stats
    assert "most_frequent" in stats
    uniform = stats["uniform"]
    assert (
        round(
            float(
                uniform.loc[uniform["metric"] == "accuracy", "mean"].iloc[0]
            ),
            2,
        )
        == _EXPECTED_DUMMY_ACCURACY
    )


def test_analyze_mlp(tmp_path):
    csv_path = tmp_path / "mlp.csv"
    csv_path.write_text(
        "experiment_name,metric.test_roc_auc,metric.test_f1_score,"
        "metric.val_auc,metric.train_loss,metric.val_loss\n"
        "tech-challenge-mlp,0.83,0.58,0.84,0.41,0.43\n"
        "tech-challenge-mlp,0.85,0.60,0.86,0.40,0.42\n"
    )
    df = load_data(str(csv_path))
    _, mlp_df, _ = split_experiments(df)
    stats = analyze_mlp(mlp_df)
    assert stats["num_runs"] == _N_MLP_RUNS
    test_stats: pd.DataFrame = stats["test_stats"]
    auc_row = test_stats[test_stats["metric"] == "test_roc_auc"]
    assert round(float(auc_row["mean"].iloc[0]), 2) == _EXPECTED_AUC
    overfitting = stats["overfitting"]
    assert isinstance(overfitting, dict)
    assert overfitting["gap_auc_mean"] is not None


def test_analyze_logistic(tmp_path):
    csv_path = tmp_path / "logistic.csv"
    csv_path.write_text(
        "experiment_name,metric.test_roc_auc,metric.test_f1_score,"
        "metric.test_accuracy,metric.cv_accuracy_mean,"
        "metric.cv_f1_mean,metric.cv_roc_auc_mean\n"
        "tech-challenge-logistic-regression,0.82,0.58,0.79,"
        "0.80,0.60,0.83\n"
        "tech-challenge-logistic-regression,0.83,0.59,0.80,"
        "0.81,0.61,0.84\n"
    )
    df = load_data(str(csv_path))
    _, _, logistic_df = split_experiments(df)
    stats = analyze_logistic(logistic_df)
    assert stats["num_runs"] == _N_LOGISTIC_RUNS
    test_stats: pd.DataFrame = stats["test_stats"]
    auc_row = test_stats[test_stats["metric"] == "test_roc_auc"]
    assert round(float(auc_row["mean"].iloc[0]), 2) == _EXPECTED_LOGISTIC_AUC
    cv_stats: pd.DataFrame = stats["cv_stats"]
    assert not cv_stats.empty
    assert "mean" in cv_stats.columns


def test_print_report(tmp_path):
    csv_path = tmp_path / "report.csv"
    csv_path.write_text(
        "experiment_name,param.strategy,"
        "metric.accuracy,metric.roc_auc,"
        "metric.test_accuracy,metric.test_roc_auc,metric.test_f1_score,"
        "metric.val_auc,metric.train_loss,metric.val_loss,"
        "metric.cv_accuracy_mean,metric.cv_f1_mean,"
        "run_name,run_id\n"
        "tech-challenge-dummy-baseline,uniform,0.5,0.5,,,,,,,"
        ",,\n"
        "tech-challenge-mlp,,,,0.79,0.83,0.59,0.84,0.41,0.42,"
        ",r1,abc\n"
        "tech-challenge-logistic-regression,,,,0.79,0.82,0.58,,,"
        "0.80,0.60,r2,def\n"
    )
    df = load_data(str(csv_path))
    dummy_df, mlp_df, logistic_df = split_experiments(df)
    dummy_stats = analyze_dummy(dummy_df)
    mlp_stats = analyze_mlp(mlp_df)
    logistic_stats = analyze_logistic(logistic_df)
    report = print_report(dummy_stats, mlp_stats, logistic_stats)
    assert "Relatorio de Analise" in report
    assert "Dummy Baseline" in report
    assert "Logistic Regression" in report
    assert "MLP" in report
    assert "Overfitting" in report

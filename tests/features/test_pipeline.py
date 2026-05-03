"""Testes para o pipeline sklearn de Logistic Regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.pipeline import build_logistic_pipeline
from src.training.metrics import compute_binary_classification_metrics


def _make_mixed_df(n_samples: int = 100) -> pd.DataFrame:
    """Cria DataFrame sintetico com colunas numericas e categoricas."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n_samples),
            "MonthlyCharges": rng.uniform(20, 120, n_samples),
            "TotalCharges": rng.uniform(0, 8000, n_samples),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n_samples
            ),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n_samples,
            ),
            "gender": rng.choice(["Female", "Male"], n_samples),
        }
    )
    return df


def _make_binary_target(n_samples: int) -> np.ndarray:
    """Cria target binario com ~30% de positivos."""
    y = np.zeros(n_samples, dtype=np.float64)
    y[: n_samples // 3] = 1.0
    rng = np.random.default_rng(42)
    rng.shuffle(y)
    return y


def test_build_logistic_pipeline_returns_pipeline() -> None:
    """build_logistic_pipeline deve retornar ImbPipeline."""
    pipeline = build_logistic_pipeline(max_iter=100, random_seed=42)
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_pipeline_fit_predict_with_smote() -> None:
    """Pipeline com SMOTE deve treinar e prever em DataFrame misto."""
    X = _make_mixed_df(100)
    y = _make_binary_target(100)

    pipeline = build_logistic_pipeline(
        max_iter=500, random_seed=42, use_smote=True
    )
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    assert len(y_pred) == len(y)
    assert len(y_proba) == len(y)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_pipeline_fit_predict_without_smote() -> None:
    """Pipeline sem SMOTE deve treinar e prever em DataFrame misto."""
    X = _make_mixed_df(100)
    y = _make_binary_target(100)

    pipeline = build_logistic_pipeline(
        max_iter=500, random_seed=42, use_smote=False
    )
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    assert len(y_pred) == len(y)
    assert len(y_proba) == len(y)


def test_pipeline_handles_missing_values() -> None:
    """Pipeline deve lidar com valores ausentes via SimpleImputer."""
    X = _make_mixed_df(100)
    X.loc[0, "MonthlyCharges"] = np.nan
    X.loc[1, "Contract"] = np.nan
    y = _make_binary_target(100)

    pipeline = build_logistic_pipeline(max_iter=500, random_seed=42)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    assert len(y_pred) == len(y)


def test_pipeline_metrics_on_split_data() -> None:
    """Metricas em hold-out devem estar no intervalo [0, 1]."""
    X = _make_mixed_df(100)
    y = _make_binary_target(100)

    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    pipeline = build_logistic_pipeline(max_iter=500, random_seed=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_binary_classification_metrics(
        y_test, y_pred, y_proba, positive_label=None
    )

    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} fora do intervalo [0, 1]"


def test_pipeline_feature_names_out() -> None:
    """Pipeline deve expor nomes das features apos encoding."""
    X = _make_mixed_df(100)
    y = _make_binary_target(100)

    pipeline = build_logistic_pipeline(max_iter=100, random_seed=42)
    pipeline.fit(X, y)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    names_list = list(feature_names)

    _min_expected = 6
    assert len(names_list) > _min_expected
    assert all(isinstance(n, str) for n in names_list)


def test_pipeline_is_deterministic() -> None:
    """Pipeline com mesma seed deve produzir resultados identicos."""
    X = _make_mixed_df(100)
    y = _make_binary_target(100)

    pipeline_1 = build_logistic_pipeline(max_iter=500, random_seed=42)
    pipeline_1.fit(X, y)
    pred_1 = pipeline_1.predict_proba(X)[:, 1]

    pipeline_2 = build_logistic_pipeline(max_iter=500, random_seed=42)
    pipeline_2.fit(X, y)
    pred_2 = pipeline_2.predict_proba(X)[:, 1]

    np.testing.assert_array_almost_equal(pred_1, pred_2)

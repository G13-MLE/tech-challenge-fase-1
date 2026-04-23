"""Testes para o modulo de preprocessamento MLP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    encode_target,
    mlp_preprocess_data,
    one_hot_encode,
    split_features_target,
)


def _make_clean_df() -> pd.DataFrame:
    """Cria DataFrame minimo limpo (saida de clean_telco_data)."""
    return pd.DataFrame(
        {
            "tenure": [1, 24, 72],
            "MonthlyCharges": [29.85, 56.95, 89.10],
            "TotalCharges": [29.85, 1366.80, 6415.20],
            "Churn": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
        }
    )


def test_encode_target_maps_yes_to_1() -> None:
    df = _make_clean_df()
    result = encode_target(df)
    assert result["Churn"].iloc[1] == 1


def test_encode_target_maps_no_to_0() -> None:
    df = _make_clean_df()
    result = encode_target(df)
    assert result["Churn"].iloc[0] == 0


def test_one_hot_encode_creates_dummy_columns() -> None:
    df = _make_clean_df()
    result = one_hot_encode(df, categorical_cols=["Contract"])
    assert "Contract_One year" in result.columns
    assert "Contract_Two year" in result.columns


def test_one_hot_encode_drops_first_category() -> None:
    df = _make_clean_df()
    result = one_hot_encode(df, categorical_cols=["Contract"])
    assert "Contract_Month-to-month" not in result.columns


def test_one_hot_encode_returns_copy_when_no_categorical() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = one_hot_encode(df)
    pd.testing.assert_frame_equal(result, df)


def test_split_features_target_separates_x_y() -> None:
    df = _make_clean_df()
    df = encode_target(df)
    X, y, features = split_features_target(df)
    assert "Churn" not in X.columns
    assert list(y) == [0, 1, 0]
    assert features == list(X.columns)


def test_mlp_preprocess_data_returns_numpy_arrays() -> None:
    df = _make_clean_df()
    X, y, _features, _df_enc = mlp_preprocess_data(df)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == np.float64
    assert y.dtype == np.float64


def test_mlp_preprocess_data_target_is_binary() -> None:
    df = _make_clean_df()
    _X, y, _features, _df_enc = mlp_preprocess_data(df)
    assert set(np.unique(y)) == {0.0, 1.0}


def test_mlp_preprocess_data_features_are_numeric() -> None:
    df = _make_clean_df()
    X, _y, features, _df_enc = mlp_preprocess_data(df)
    assert len(features) == X.shape[1]
    assert all(isinstance(f, str) for f in features)

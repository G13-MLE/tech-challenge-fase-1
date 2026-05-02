"""Testes para o modulo de inferencia da API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.api.inference import (
    ChurnPredictor,
    _infer_mlp_config,  # noqa: PLC2701
    _PredictorHolder,  # noqa: PLC2701
    _prepare_dataframe,  # noqa: PLC2701
    _preprocess_for_inference,  # noqa: PLC2701
    get_predictor,
)

_INPUT_DIM = 4
_HIDDEN_DIM_1 = 8
_HIDDEN_DIM_2 = 2
_BATCH_SIZE = 1


class TestInferMLPConfig:
    """Testes para inferencia de configuracao a partir de state_dict."""

    @staticmethod
    def test_infer_basic_config() -> None:
        state_dict = {
            "hidden_layers.0.weight": torch.randn(_HIDDEN_DIM_1, _INPUT_DIM),
            "hidden_layers.0.bias": torch.randn(_HIDDEN_DIM_1),
            "hidden_layers.2.weight": torch.randn(
                _HIDDEN_DIM_2, _HIDDEN_DIM_1
            ),
            "hidden_layers.2.bias": torch.randn(_HIDDEN_DIM_2),
            "output_layer.weight": torch.randn(_BATCH_SIZE, _HIDDEN_DIM_2),
            "output_layer.bias": torch.randn(_BATCH_SIZE),
        }
        config = _infer_mlp_config(state_dict)
        assert config.input_dim == _INPUT_DIM
        assert config.hidden_dims == (_HIDDEN_DIM_1, _HIDDEN_DIM_2)
        assert config.use_batch_norm is False

    @staticmethod
    def test_infer_with_batch_norm() -> None:
        state_dict = {
            "hidden_layers.0.weight": torch.randn(_HIDDEN_DIM_1, _INPUT_DIM),
            "hidden_layers.0.bias": torch.randn(_HIDDEN_DIM_1),
            "hidden_layers.1.weight": torch.randn(_HIDDEN_DIM_1),
            "hidden_layers.1.bias": torch.randn(_HIDDEN_DIM_1),
            "hidden_layers.1.running_mean": torch.randn(_HIDDEN_DIM_1),
            "hidden_layers.1.running_var": torch.randn(_HIDDEN_DIM_1),
            "hidden_layers.4.weight": torch.randn(
                _HIDDEN_DIM_2, _HIDDEN_DIM_1
            ),
            "hidden_layers.4.bias": torch.randn(_HIDDEN_DIM_2),
            "output_layer.weight": torch.randn(_BATCH_SIZE, _HIDDEN_DIM_2),
            "output_layer.bias": torch.randn(_BATCH_SIZE),
        }
        config = _infer_mlp_config(state_dict)
        assert config.input_dim == _INPUT_DIM
        assert config.hidden_dims == (_HIDDEN_DIM_1, _HIDDEN_DIM_2)
        assert config.use_batch_norm is True


class TestPrepareDataFrame:
    """Testes para montagem e limpeza do DataFrame de inferencia."""

    @staticmethod
    def _base_customer_data() -> dict[str, object]:
        return {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 50.0,
            "Churn": "No",
        }

    @staticmethod
    def test_prepare_complete_data() -> None:
        data = TestPrepareDataFrame._base_customer_data()
        data["TotalCharges"] = 600.0
        df = _prepare_dataframe(data)
        assert len(df) == _BATCH_SIZE
        expected_total = 600.0
        assert df["TotalCharges"].iloc[0] == expected_total

    @staticmethod
    def test_prepare_missing_total_charges() -> None:
        data = TestPrepareDataFrame._base_customer_data()
        data["tenure"] = 10
        data["MonthlyCharges"] = 70.0
        df = _prepare_dataframe(data)
        assert len(df) == _BATCH_SIZE
        expected_total = 700.0
        assert df["TotalCharges"].iloc[0] == expected_total

    @staticmethod
    def test_prepare_zero_tenure_missing_total() -> None:
        data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 29.0,
            "Churn": "No",
        }
        df = _prepare_dataframe(data)
        assert len(df) == _BATCH_SIZE
        expected_total = 0.0
        assert df["TotalCharges"].iloc[0] == expected_total


class TestPreprocessForInference:
    """Testes para preprocessamento consistente de inferencia."""

    @staticmethod
    def test_generates_all_features() -> None:
        feature_names = [
            "gender_Male",
            "SeniorCitizen",
            "Partner_No",
            "tenure",
            "PhoneService_Yes",
            "Contract_One year",
            "Contract_Two year",
            "MonthlyCharges",
            "TotalCharges",
        ]
        df = pd.DataFrame(
            [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "tenure": 5,
                    "PhoneService": "No",
                    "Contract": "Month-to-month",
                    "MonthlyCharges": 30.0,
                    "TotalCharges": 150.0,
                }
            ]
        )
        X = _preprocess_for_inference(df, feature_names)
        assert X.shape == (_BATCH_SIZE, len(feature_names))
        assert X[0, feature_names.index("gender_Male")] == 0.0
        assert X[0, feature_names.index("Partner_No")] == 0.0
        assert X[0, feature_names.index("PhoneService_Yes")] == 0.0
        assert X[0, feature_names.index("Contract_One year")] == 0.0
        assert X[0, feature_names.index("Contract_Two year")] == 0.0

    @staticmethod
    def test_drop_first_category() -> None:
        feature_names = ["gender_Male", "Partner_Yes", "tenure"]
        df = pd.DataFrame(
            [
                {
                    "gender": "Male",
                    "Partner": "Yes",
                    "tenure": 10,
                }
            ]
        )
        X = _preprocess_for_inference(df, feature_names)
        assert X[0, 0] == 1.0  # gender_Male
        assert X[0, 1] == 1.0  # Partner_Yes

    @staticmethod
    def test_missing_features_filled_with_zero() -> None:
        feature_names = [
            "gender_Male",
            "Missing_Feature",
            "tenure",
        ]
        df = pd.DataFrame([{"gender": "Female", "tenure": 3}])
        X = _preprocess_for_inference(df, feature_names)
        assert X[0, 1] == 0.0  # Missing_Feature preenchido


class TestChurnPredictor:
    """Testes para o wrapper ChurnPredictor."""

    @staticmethod
    @patch("src.api.inference.torch.load")
    @patch("src.api.inference.load_scaler")
    @patch(
        "src.api.inference._load_feature_names",
        return_value=["feat_a", "feat_b"],
    )
    def test_load_and_predict(
        mock_load_fn, mock_load_scaler, mock_torch_load
    ) -> None:
        mock_state_dict = {
            "hidden_layers.0.weight": torch.randn(4, 3),
            "hidden_layers.0.bias": torch.randn(4),
            "output_layer.weight": torch.randn(1, 4),
            "output_layer.bias": torch.randn(1),
        }
        mock_torch_load.return_value = {"model_state_dict": mock_state_dict}

        mock_scaler = MagicMock()
        mock_scaler.transform = MagicMock(
            return_value=np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        )
        mock_load_scaler.return_value = mock_scaler

        predictor = ChurnPredictor(
            model_path="models/churn_mlp_best.pt",
            scaler_path="models/scaler.pkl",
            feature_names_path="models/feature_names.json",
        )

        with patch.object(Path, "exists", return_value=True):
            predictor.load()

        assert predictor._model is not None
        assert predictor._scaler is not None

        predictor._model.eval()
        with torch.no_grad():
            prob = predictor.predict(
                {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Credit card (automatic)",
                    "MonthlyCharges": 29.0,
                    "TotalCharges": 29.0,
                    "Churn": "No",
                }
            )

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    @staticmethod
    @patch("src.api.inference.torch.load")
    @patch("src.api.inference.load_scaler")
    @patch(
        "src.api.inference._load_feature_names",
        return_value=["feat_a"],
    )
    def test_predictor_singleton(
        mock_load_fn, mock_load_scaler, mock_torch
    ) -> None:
        previous = _PredictorHolder.instance
        _PredictorHolder.instance = None

        p1 = get_predictor()
        p2 = get_predictor()
        assert p1 is p2

        _PredictorHolder.instance = previous


class TestPredictorErrors:
    """Testes para cenarios de erro no predictor."""

    @staticmethod
    def test_model_not_found() -> None:
        predictor = ChurnPredictor(
            model_path="models/inexistente.pt",
            scaler_path="models/scaler.pkl",
        )
        with pytest.raises(FileNotFoundError):
            predictor.load()

    @staticmethod
    def test_scaler_not_found() -> None:
        predictor = ChurnPredictor(
            model_path="models/churn_mlp_best.pt",
            scaler_path="models/inexistente.pkl",
        )
        with pytest.raises(FileNotFoundError):
            predictor.load()

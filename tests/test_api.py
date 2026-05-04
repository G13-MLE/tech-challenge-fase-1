from __future__ import annotations

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def base_payload() -> dict[str, object]:
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
    }


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


@patch("src.api.main.predict_single", return_value=0.85)
def test_predict_endpoint_high_churn(mock_predict: object) -> None:
    payload = base_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["churn_probability"] == 0.85  # noqa: PLR2004
    assert data["churn_prediction"] is True
    mock_predict.assert_called_once()


@patch("src.api.main.predict_single", return_value=0.15)
def test_predict_endpoint_low_churn(mock_predict: object) -> None:
    payload = base_payload()
    payload["tenure"] = 15
    payload["Contract"] = "One year"
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["churn_probability"] == 0.15  # noqa: PLR2004
    assert data["churn_prediction"] is False


def test_predict_endpoint_validation_error() -> None:
    invalid_payload = {
        "customerID": "7590-VHVEG",
        "MonthlyCharges": "29.85",
        "Contract": "Month-to-month",
        "ExtraField": "Not allowed",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@patch("src.api.main.predict_single", return_value=0.6)
def test_predict_endpoint_with_total_charges(mock_predict: object) -> None:
    payload = base_payload()
    payload["TotalCharges"] = 100.0
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["churn_probability"] == 0.6  # noqa: PLR2004
    assert data["churn_prediction"] is True

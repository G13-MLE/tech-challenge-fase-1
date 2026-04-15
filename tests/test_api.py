from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_high_churn() -> None:
    payload = {
        "customerID": "7590-VHVEG",
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "Contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    expected_prob = 0.85
    assert data["churn_probability"] == expected_prob
    assert data["churn_prediction"] is True


def test_predict_endpoint_low_churn() -> None:
    payload = {
        "customerID": "1234-ABCDE",
        "tenure": 15,
        "MonthlyCharges": 50.00,
        "Contract": "One year",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    expected_prob = 0.15
    assert data["churn_probability"] == expected_prob
    assert data["churn_prediction"] is False


def test_predict_endpoint_validation_error() -> None:
    # Missing required field 'tenure' and wrong type for 'MonthlyCharges'
    invalid_payload = {
        "customerID": "7590-VHVEG",
        "MonthlyCharges": "29.85",  # should be float
        "Contract": "Month-to-month",
        "ExtraField": "Not allowed",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

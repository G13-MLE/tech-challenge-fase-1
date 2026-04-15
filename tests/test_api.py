from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint() -> None:
    payload = {
        "customerID": "7590-VHVEG",
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "Contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert isinstance(data["churn_probability"], float)
    assert isinstance(data["churn_prediction"], bool)


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

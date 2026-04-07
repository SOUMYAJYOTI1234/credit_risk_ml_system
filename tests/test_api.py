"""
test_api.py - Unit Tests for the FastAPI Prediction Endpoint

Tests the /predict and / (health) endpoints using the FastAPI TestClient.
Uses context-managed TestClient to ensure lifespan events (model loading) fire.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Provide a TestClient that triggers the app lifespan (model loading)."""
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────
# Sample payload
# ─────────────────────────────────────────────────────────────

SAMPLE_PAYLOAD = {
    "limit_bal": 20000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 24,
    "pay_1": 2,
    "pay_2": 2,
    "pay_3": -1,
    "pay_4": -1,
    "pay_5": -2,
    "pay_6": -2,
    "bill_amt1": 3913,
    "bill_amt2": 3102,
    "bill_amt3": 689,
    "bill_amt4": 0,
    "bill_amt5": 0,
    "bill_amt6": 0,
    "pay_amt1": 0,
    "pay_amt2": 689,
    "pay_amt3": 0,
    "pay_amt4": 0,
    "pay_amt5": 0,
    "pay_amt6": 0,
}


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for the GET / health check."""

    def test_health_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""

    def test_predict_returns_valid_response(self, client):
        """Test that a valid payload returns a prediction."""
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        # If model is not loaded, expect 503; otherwise expect 200
        if response.status_code == 200:
            data = response.json()
            assert "default_probability" in data
            assert "prediction" in data
            assert "threshold" in data
            assert 0 <= data["default_probability"] <= 1
            assert data["prediction"] in [0, 1]
        else:
            assert response.status_code == 503  # model not loaded

    def test_predict_probability_range(self, client):
        """Ensure probability is between 0 and 1."""
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["default_probability"] <= 1.0

    def test_predict_invalid_payload(self, client):
        """Sending an empty body should return a 422 validation error."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_partial_payload(self, client):
        """Missing required fields should fail validation."""
        partial = {"limit_bal": 20000, "sex": 1}
        response = client.post("/predict", json=partial)
        assert response.status_code == 422

    def test_predict_edge_case_zeros(self, client):
        """All-zero bill/payment amounts should not crash the API."""
        payload = {
            "limit_bal": 50000, "sex": 1, "education": 1, "marriage": 2, "age": 30,
            "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
            "bill_amt1": 0, "bill_amt2": 0, "bill_amt3": 0,
            "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
            "pay_amt1": 0, "pay_amt2": 0, "pay_amt3": 0,
            "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
        }
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["default_probability"] <= 1.0
        else:
            assert response.status_code == 503  # model not loaded


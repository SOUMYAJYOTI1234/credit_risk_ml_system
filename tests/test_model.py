"""
test_model.py - Unit Tests for Model Loading and Prediction

Verifies that:
  • The trained model file exists and loads correctly.
  • Predictions return valid probabilities in [0, 1].
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.utils import get_models_dir
from src.features import engineer_features


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(get_models_dir(), "model.pkl")


@pytest.fixture
def sample_input() -> pd.DataFrame:
    """Create a single-row DataFrame that mimics real input data."""
    data = {
        "limit_bal": [20000],
        "sex": [2],
        "education": [2],
        "marriage": [1],
        "age": [24],
        "pay_1": [2],
        "pay_2": [2],
        "pay_3": [-1],
        "pay_4": [-1],
        "pay_5": [-2],
        "pay_6": [-2],
        "bill_amt1": [3913],
        "bill_amt2": [3102],
        "bill_amt3": [689],
        "bill_amt4": [0],
        "bill_amt5": [0],
        "bill_amt6": [0],
        "pay_amt1": [0],
        "pay_amt2": [689],
        "pay_amt3": [0],
        "pay_amt4": [0],
        "pay_amt5": [0],
        "pay_amt6": [0],
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestModelLoading:
    """Tests that the model loads correctly from disk."""

    def test_model_file_exists(self):
        """Check that models/model.pkl exists."""
        assert os.path.exists(MODEL_PATH), (
            f"Model file not found at {MODEL_PATH}. "
            "Run `python -m src.train` first."
        )

    def test_model_loads_successfully(self):
        """Check that joblib can deserialise the model."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        model = joblib.load(MODEL_PATH)
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestModelPrediction:
    """Tests that the loaded model returns valid predictions."""

    def test_predict_returns_array(self, sample_input):
        """Predictions should be a numpy array."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        model = joblib.load(MODEL_PATH)
        df = engineer_features(sample_input)
        preds = model.predict(df)
        assert isinstance(preds, np.ndarray)

    def test_predict_proba_in_range(self, sample_input):
        """Probabilities should be between 0 and 1."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        model = joblib.load(MODEL_PATH)
        df = engineer_features(sample_input)
        proba = model.predict_proba(df)

        assert proba.shape[1] == 2  # binary classification
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_single_sample(self, sample_input):
        """A single sample should produce exactly one prediction."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        model = joblib.load(MODEL_PATH)
        df = engineer_features(sample_input)
        preds = model.predict(df)
        assert len(preds) == 1
        assert preds[0] in [0, 1]

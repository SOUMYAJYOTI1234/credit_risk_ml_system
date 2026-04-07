"""
test_model.py - Unit Tests for Model Loading and Prediction

Verifies that:
  • The trained model Pipeline exists and loads correctly.
  • The Pipeline handles feature engineering + prediction end-to-end.
  • Predictions return valid probabilities in [0, 1].
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.utils import get_models_dir


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(get_models_dir(), "model.pkl")


@pytest.fixture
def sample_input() -> pd.DataFrame:
    """Create a single-row DataFrame with raw input (no engineered features).

    The Pipeline should handle feature engineering internally.
    """
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


@pytest.fixture
def edge_case_zeros() -> pd.DataFrame:
    """Edge case: all bill and payment amounts are zero."""
    data = {
        "limit_bal": [50000],
        "sex": [1],
        "education": [1],
        "marriage": [2],
        "age": [30],
        "pay_1": [0],
        "pay_2": [0],
        "pay_3": [0],
        "pay_4": [0],
        "pay_5": [0],
        "pay_6": [0],
        "bill_amt1": [0],
        "bill_amt2": [0],
        "bill_amt3": [0],
        "bill_amt4": [0],
        "bill_amt5": [0],
        "bill_amt6": [0],
        "pay_amt1": [0],
        "pay_amt2": [0],
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
    """Tests that the model Pipeline loads correctly from disk."""

    def test_model_file_exists(self):
        """Check that models/model.pkl exists (skips in CI)."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python -m src.train` first."
            )

    def test_model_loads_successfully(self):
        """Check that joblib can deserialise the Pipeline."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        model = joblib.load(MODEL_PATH)
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_is_pipeline(self):
        """The saved model should be a sklearn Pipeline, not a raw estimator."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        from sklearn.pipeline import Pipeline
        model = joblib.load(MODEL_PATH)
        assert isinstance(model, Pipeline), (
            f"Expected Pipeline, got {type(model).__name__}"
        )


class TestModelPrediction:
    """Tests that the loaded Pipeline returns valid predictions."""

    def test_pipeline_end_to_end(self, sample_input):
        """Full integration test: load Pipeline, feed raw input, get prediction."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        pipeline = joblib.load(MODEL_PATH)
        # Feed RAW input (no manual engineer_features call)
        proba = pipeline.predict_proba(sample_input)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_predict_returns_array(self, sample_input):
        """Predictions should be a numpy array."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        pipeline = joblib.load(MODEL_PATH)
        preds = pipeline.predict(sample_input)
        assert isinstance(preds, np.ndarray)

    def test_predict_proba_in_range(self, sample_input):
        """Probabilities should be between 0 and 1."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        pipeline = joblib.load(MODEL_PATH)
        proba = pipeline.predict_proba(sample_input)
        assert proba.shape[1] == 2
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_single_sample(self, sample_input):
        """A single sample should produce exactly one prediction."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        pipeline = joblib.load(MODEL_PATH)
        preds = pipeline.predict(sample_input)
        assert len(preds) == 1
        assert preds[0] in [0, 1]

    def test_predict_edge_case_zeros(self, edge_case_zeros):
        """All-zero amounts should not produce NaN probabilities."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        pipeline = joblib.load(MODEL_PATH)
        proba = pipeline.predict_proba(edge_case_zeros)
        assert not np.any(np.isnan(proba)), "NaN probability for all-zero input"
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_predict_undocumented_categoricals(self, sample_input):
        """Undocumented categorical values (e.g., education=0) should be handled."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not available")
        pipeline = joblib.load(MODEL_PATH)
        # education=0 is undocumented in the UCI dataset
        bad_input = sample_input.copy()
        bad_input["education"] = 0
        bad_input["marriage"] = 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            proba = pipeline.predict_proba(bad_input)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0


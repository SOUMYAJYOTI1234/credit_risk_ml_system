"""
main.py - FastAPI Application for Credit Risk Prediction

The loaded model.pkl is a self-contained sklearn Pipeline
(CreditFeatureTransformer → [optional Scaler] → Classifier).
No separate feature engineering call is needed.

Endpoints:
  GET  /         → Health check
  POST /predict  → Predict default probability for a single applicant
"""

import os
import logging
from contextlib import asynccontextmanager

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CreditApplicationRequest, PredictionResponse, HealthResponse
from src.utils import get_models_dir, load_json

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────

MODEL = None   # will hold the full Pipeline
THRESHOLD = 0.5
VERSION = "2.0.0"

MODELS_DIR = get_models_dir()
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")


# ─────────────────────────────────────────────────────────────
# Lifespan – model + threshold loading
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Pipeline and optimized threshold on startup."""
    global MODEL, THRESHOLD

    # Load Pipeline
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        logger.info("Pipeline loaded from %s", MODEL_PATH)
    else:
        logger.warning(
            "Model file not found at %s — /predict will be unavailable",
            MODEL_PATH,
        )

    # Load optimized threshold
    if os.path.exists(THRESHOLD_PATH):
        threshold_info = load_json(THRESHOLD_PATH)
        THRESHOLD = threshold_info.get("threshold", 0.5)
        logger.info(
            "Loaded optimized threshold: %.4f (strategy=%s)",
            THRESHOLD, threshold_info.get("strategy", "unknown"),
        )
    else:
        logger.info("No threshold.json found — using default threshold=%.2f", THRESHOLD)

    yield
    MODEL = None


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Predicts the probability that a credit-card holder will default "
        "on the next monthly payment, based on 23 demographic and "
        "transactional features from the UCI Credit Card Default dataset. "
        "The model is a self-contained sklearn Pipeline that handles "
        "feature engineering internally."
    ),
    version=VERSION,
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Health-check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        version=VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: CreditApplicationRequest):
    """Predict the default probability for a single credit-card applicant.

    The request body should contain all 23 original features.
    Feature engineering is handled automatically by the Pipeline.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first.",
        )

    # Convert request to DataFrame (single row)
    input_dict = request.model_dump()
    df = pd.DataFrame([input_dict])

    # Predict using the full Pipeline (features + model)
    try:
        proba = float(MODEL.predict_proba(df)[:, 1][0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Post-prediction validation
    if np.isnan(proba):
        logger.error("Model produced NaN probability for input: %s", input_dict)
        raise HTTPException(
            status_code=500,
            detail="Model produced an invalid probability. Check input values.",
        )

    prediction = int(proba >= THRESHOLD)

    return PredictionResponse(
        default_probability=round(proba, 6),
        prediction=prediction,
        threshold=THRESHOLD,
    )

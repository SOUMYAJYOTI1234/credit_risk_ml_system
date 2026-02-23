"""
main.py - FastAPI Application for Credit Risk Prediction

Endpoints:
  GET  /         → Health check
  POST /predict  → Predict default probability for a single applicant
"""

import os
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CreditApplicationRequest, PredictionResponse, HealthResponse
from src.features import engineer_features
from src.utils import get_models_dir

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────

MODEL = None
MODEL_PATH = os.path.join(get_models_dir(), "model.pkl")
THRESHOLD = 0.5  # default threshold; can be updated from reports
VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────
# Lifespan – model loading
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found at %s — /predict will be unavailable", MODEL_PATH)
    yield
    # Cleanup (if needed)
    MODEL = None


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Predicts the probability that a credit-card holder will default "
        "on the next monthly payment, based on 23 demographic and "
        "transactional features from the UCI Credit Card Default dataset."
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
    Feature engineering (avg_bill_amt, credit_utilization, etc.) is applied
    automatically before prediction.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first.",
        )

    # Convert request to DataFrame (single row)
    input_dict = request.model_dump()
    df = pd.DataFrame([input_dict])

    # Apply feature engineering
    df = engineer_features(df)

    # Predict
    try:
        proba = float(MODEL.predict_proba(df)[:, 1][0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    prediction = int(proba >= THRESHOLD)

    return PredictionResponse(
        default_probability=round(proba, 6),
        prediction=prediction,
        threshold=THRESHOLD,
    )

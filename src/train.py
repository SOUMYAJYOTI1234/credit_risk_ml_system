"""
train.py - Model Training Module

Trains Logistic Regression, Random Forest, and XGBoost on the
credit-card default dataset using sklearn Pipelines that encapsulate
feature engineering + optional scaling + model.

The saved model.pkl is a self-contained Pipeline — no separate
feature engineering step is needed at inference time.
"""

import os
import glob
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_loader import load_cleaned_data
from src.evaluate import evaluate_model, find_optimal_threshold
from src.features import CreditFeatureTransformer
from src.utils import (
    get_models_dir, setup_logging, save_json,
    get_reports_dir, timestamp_str,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

TARGET_COL = "default"
MAX_MODEL_VERSIONS = 5  # keep this many historical model files


# ─────────────────────────────────────────────────────────────
# Pipeline definitions
# ─────────────────────────────────────────────────────────────


def get_pipelines() -> Dict[str, Pipeline]:
    """Return named pipelines: feature transformer + (optional scaler) + model.

    LogisticRegression gets a StandardScaler because lbfgs is sensitive
    to feature scale (limit_bal ~ 10^5 vs age ~ 10^1).
    Tree-based models don't need scaling.
    """

    return {
        "LogisticRegression": Pipeline([
            ("features", CreditFeatureTransformer()),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("features", CreditFeatureTransformer()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("features", CreditFeatureTransformer()),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=3.5,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# ─────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> np.ndarray:
    """Run stratified k-fold cross-validation and return per-fold scores.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn Pipeline (features + model).
    X : pd.DataFrame
        Training features (raw — will be transformed by the Pipeline).
    y : pd.Series
        Training labels.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric.

    Returns
    -------
    np.ndarray
        Array of scores for each fold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return scores


# ─────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """Train all candidate pipelines and return fitted pipelines + CV results.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw training features (pre feature-engineering).
    y_train : pd.Series
        Training labels.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    fitted_pipelines : dict  {name: fitted Pipeline}
    cv_results :       dict  {name: {"mean_auc": ..., "std_auc": ...}}
    """
    pipelines = get_pipelines()
    fitted_pipelines: Dict[str, Pipeline] = {}
    cv_results: Dict[str, Dict[str, float]] = {}

    for name, pipe in pipelines.items():
        logger.info("Training %s …", name)

        # Cross-validation (each fold fits features + model from scratch)
        scores = cross_validate_model(pipe, X_train, y_train, cv=cv)
        mean_auc = float(scores.mean())
        std_auc = float(scores.std())
        logger.info("%s  CV ROC-AUC: %.4f ± %.4f", name, mean_auc, std_auc)

        # Fit on the full training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        cv_results[name] = {"mean_auc": mean_auc, "std_auc": std_auc}

    return fitted_pipelines, cv_results


def select_best_model(
    cv_results: Dict[str, Dict[str, float]],
    fitted_pipelines: Dict[str, Pipeline],
) -> Tuple[str, Pipeline]:
    """Select the pipeline with the highest mean CV ROC-AUC."""
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
    logger.info(
        "Best model: %s (AUC=%.4f)",
        best_name, cv_results[best_name]["mean_auc"],
    )
    return best_name, fitted_pipelines[best_name]


# ─────────────────────────────────────────────────────────────
# Model saving with versioning
# ─────────────────────────────────────────────────────────────


def save_model(pipeline: Pipeline, filename: str = "model.pkl") -> str:
    """Save a trained pipeline with versioning.

    Saves two files:
      - models/model.pkl          (latest, used by the API)
      - models/model_<timestamp>.pkl  (versioned backup)

    Old versions beyond MAX_MODEL_VERSIONS are cleaned up.
    """
    models_dir = get_models_dir()

    # Save as latest
    latest_path = os.path.join(models_dir, filename)
    joblib.dump(pipeline, latest_path)
    logger.info("Pipeline saved to %s", latest_path)

    # Save versioned copy
    ts = timestamp_str()
    versioned_name = f"model_{ts}.pkl"
    versioned_path = os.path.join(models_dir, versioned_name)
    joblib.dump(pipeline, versioned_path)
    logger.info("Versioned copy → %s", versioned_path)

    # Cleanup old versions (keep last N)
    pattern = os.path.join(models_dir, "model_*.pkl")
    versions = sorted(glob.glob(pattern))
    if len(versions) > MAX_MODEL_VERSIONS:
        for old in versions[:-MAX_MODEL_VERSIONS]:
            os.remove(old)
            logger.info("Removed old model version: %s", old)

    return latest_path


# ─────────────────────────────────────────────────────────────
# Main training entrypoint
# ─────────────────────────────────────────────────────────────


def run_training_pipeline() -> Tuple[str, Pipeline]:
    """End-to-end training pipeline.

    Steps:
        1. Load and clean data
        2. Split data (raw — feature engineering is inside the Pipeline)
        3. Train & cross-validate all pipelines
        4. Select best pipeline
        5. Save pipeline + CV results (with versioning)
        6. Evaluate on test set (metrics + plots)
        7. Optimize decision threshold and save it

    Returns
    -------
    best_name : str
    best_pipeline : Pipeline
    """
    setup_logging()

    # 1. Load data
    df = load_cleaned_data()
    logger.info("Loaded data: %s", df.shape)

    # 2. Split (raw features — Pipeline handles engineering internally)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(
        "Split → train: %d, test: %d (%.1f%% positive in test)",
        len(X_train), len(X_test), 100 * y_test.mean(),
    )

    # 3. Train all pipelines
    fitted_pipelines, cv_results = train_all_models(X_train, y_train)

    # 4. Select best
    best_name, best_pipeline = select_best_model(cv_results, fitted_pipelines)

    # 5. Save artefacts (with versioning)
    save_model(best_pipeline)
    save_json(cv_results, os.path.join(get_reports_dir(), "cv_results.json"))

    # Save test data for downstream use
    test_path = os.path.join(get_models_dir(), "test_data.pkl")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    logger.info("Test data saved to %s", test_path)

    # 6. Evaluate best pipeline on test set → metrics + plots
    logger.info("Evaluating %s on the test set …", best_name)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = best_pipeline.predict(X_test)

    from src.evaluate import compute_metrics
    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    logger.info(
        "Test metrics → AUC: %.4f | F1: %.4f | AP: %.4f",
        metrics["roc_auc"], metrics["f1_score"], metrics["average_precision"],
    )

    # Save evaluation plots and metrics
    evaluate_model(best_pipeline, X_test, y_test, model_name=best_name)

    # 7. Optimize decision threshold (F1-based) and save
    optimal_threshold = find_optimal_threshold(y_test.values, y_proba, strategy="f1")
    threshold_info = {
        "threshold": optimal_threshold,
        "strategy": "f1",
        "model_name": best_name,
    }
    threshold_path = os.path.join(get_models_dir(), "threshold.json")
    save_json(threshold_info, threshold_path)
    logger.info("Optimal threshold (%.4f) saved to %s", optimal_threshold, threshold_path)

    return best_name, best_pipeline


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best_name, _ = run_training_pipeline()
    print(f"\n✅ Training complete. Best model: {best_name}")

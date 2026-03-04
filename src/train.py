"""
train.py - Model Training Module

Trains Logistic Regression, Random Forest, and XGBoost on the
credit-card default dataset and saves the best model to disk.
"""

import os
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from src.data_loader import load_cleaned_data
from src.features import engineer_features
from src.utils import split_data, get_models_dir, setup_logging, save_json, get_reports_dir

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────


def get_models() -> Dict[str, Any]:
    """Return a dictionary of named model instances with sensible defaults."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3.5,  # approx ratio of neg/pos
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> np.ndarray:
    """Run stratified k-fold cross-validation and return per-fold scores.

    Parameters
    ----------
    model : estimator
        Sklearn-compatible estimator.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training labels.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric (e.g. 'roc_auc', 'f1', 'precision', 'recall').

    Returns
    -------
    np.ndarray
        Array of scores for each fold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return scores


# ─────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train all candidate models and return fitted estimators + CV results.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    fitted_models : dict
        {name: fitted_estimator}
    cv_results : dict
        {name: {"mean_auc": ..., "std_auc": ...}}
    """
    models = get_models()
    fitted_models: Dict[str, Any] = {}
    cv_results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        logger.info("Training %s …", name)

        # Cross-validation
        scores = cross_validate_model(model, X_train, y_train, cv=cv)
        mean_auc = float(scores.mean())
        std_auc = float(scores.std())
        logger.info(
            "%s  CV ROC-AUC: %.4f ± %.4f",
            name, mean_auc, std_auc,
        )

        # Fit on the full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model
        cv_results[name] = {"mean_auc": mean_auc, "std_auc": std_auc}

    return fitted_models, cv_results


def select_best_model(
    cv_results: Dict[str, Dict[str, float]],
    fitted_models: Dict[str, Any],
) -> Tuple[str, Any]:
    """Select the model with the highest mean CV ROC-AUC.

    Returns
    -------
    best_name : str
    best_model : estimator
    """
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
    logger.info(
        "Best model: %s (AUC=%.4f)",
        best_name,
        cv_results[best_name]["mean_auc"],
    )
    return best_name, fitted_models[best_name]


def save_model(model: Any, filename: str = "model.pkl") -> str:
    """Persist a trained model to the models/ directory using joblib.

    Parameters
    ----------
    model : estimator
        Fitted sklearn/xgboost model.
    filename : str
        File name for the pickle artefact.

    Returns
    -------
    str
        Path to the saved model file.
    """
    models_dir = get_models_dir()
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)
    return filepath


# ─────────────────────────────────────────────────────────────
# Main training entrypoint
# ─────────────────────────────────────────────────────────────

def run_training_pipeline() -> Tuple[str, Any]:
    """End-to-end training pipeline.

    Steps:
        1. Load and clean data
        2. Engineer features
        3. Split data
        4. Train & cross-validate all models
        5. Select best model
        6. Save best model + CV results

    Returns
    -------
    best_name : str
    best_model : estimator
    """
    setup_logging()

    # 1. Load data
    df = load_cleaned_data()
    logger.info("Loaded data: %s", df.shape)

    # 2. Engineer features
    df = engineer_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Train all models
    fitted_models, cv_results = train_all_models(X_train, y_train)

    # 5. Select best
    best_name, best_model = select_best_model(cv_results, fitted_models)

    # 6. Save artefacts
    save_model(best_model)
    save_json(cv_results, os.path.join(get_reports_dir(), "cv_results.json"))

    # Also save the test set for downstream evaluation
    test_path = os.path.join(get_models_dir(), "test_data.pkl")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    logger.info("Test data saved to %s", test_path)

    return best_name, best_model


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best_name, _ = run_training_pipeline()
    print(f"\n✅ Training complete. Best model: {best_name}")

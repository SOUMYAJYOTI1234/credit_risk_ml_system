"""
utils.py - Shared Utility Functions

Common helpers used across the project for logging, path resolution,
configuration, and data splitting.
"""

import os
import logging
import json
from datetime import datetime
from typing import Tuple, Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_root() -> str:
    """Return the absolute path to the project root."""
    return PROJECT_ROOT


def get_data_dir() -> str:
    """Return the absolute path to the data/ directory."""
    return os.path.join(PROJECT_ROOT, "data")


def get_models_dir() -> str:
    """Return the absolute path to the models/ directory."""
    path = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(path, exist_ok=True)
    return path


def get_reports_dir() -> str:
    """Return the absolute path to the reports/ directory (created on demand)."""
    path = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging with a timestamp format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────────────────────

TARGET_COL = "default"


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must include the target column).
    target : str
        Name of the target column.
    test_size : float
        Fraction for the test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(
        "Split complete → train: %d, test: %d (%.1f%% positive in test)",
        len(X_train),
        len(X_test),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy types to native Python types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_convert)
    logging.info("Saved JSON to %s", filepath)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON file and return the dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Timestamp helper
# ─────────────────────────────────────────────────────────────

def timestamp_str() -> str:
    """Return a filesystem-safe timestamp string for versioning artefacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

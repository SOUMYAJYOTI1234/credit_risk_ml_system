"""
evaluate.py - Model Evaluation Module

Provides functions for:
  • Classification report (precision, recall, F1)
  • Confusion matrix
  • ROC-AUC + ROC curve plot
  • Precision-Recall curve + threshold optimization
"""

import os
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

from src.utils import get_reports_dir, save_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary with AUC, precision, recall, F1 and classification report.
    """
    auc = roc_auc_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_proba)

    metrics = {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Full evaluation of a fitted model on a held-out test set.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_name : str
        Label used in plot titles and file names.
    save_plots : bool
        Whether to save the ROC and confusion matrix plots.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    logger.info(
        "%s | AUC: %.4f | F1: %.4f | AP: %.4f",
        model_name,
        metrics["roc_auc"],
        metrics["f1_score"],
        metrics["average_precision"],
    )

    if save_plots:
        reports_dir = get_reports_dir()
        plot_confusion_matrix(y_test.values, y_pred, model_name, reports_dir)
        plot_roc_curve(y_test.values, y_proba, model_name, reports_dir)
        plot_precision_recall_curve(y_test.values, y_proba, model_name, reports_dir)
        save_json(metrics, os.path.join(reports_dir, f"{model_name}_metrics.json"))

    return metrics


# ─────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{title}_confusion_matrix.png")
        fig.savefig(path, dpi=150)
        logger.info("Saved confusion matrix to %s", path)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """Plot and optionally save a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{title}_roc_curve.png")
        fig.savefig(path, dpi=150)
        logger.info("Saved ROC curve to %s", path)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """Plot the precision-recall curve with threshold annotations."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {title}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{title}_precision_recall_curve.png")
        fig.savefig(path, dpi=150)
        logger.info("Saved PR curve to %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
) -> float:
    """Select the decision threshold that maximises the chosen metric.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_proba : array-like
        Predicted probabilities.
    strategy : str
        One of 'f1' (maximise F1) or 'recall_80' (≥ 80% recall).

    Returns
    -------
    float
        Optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if strategy == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        logger.info(
            "Optimal F1 threshold: %.4f (F1=%.4f, P=%.4f, R=%.4f)",
            best_threshold, f1_scores[best_idx],
            precision[best_idx], recall[best_idx],
        )
    elif strategy == "recall_80":
        # Find the highest threshold that still achieves ≥ 80% recall
        valid = recall >= 0.80
        if valid.any():
            idx = np.where(valid)[0][-1]
            best_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        else:
            best_threshold = 0.5
        logger.info("Threshold for ≥80%% recall: %.4f", best_threshold)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return best_threshold

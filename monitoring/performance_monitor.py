"""
performance_monitor.py - Model Performance Monitoring

Tracks model performance (AUC) over time and triggers retraining
alerts when the metric drops below a configurable threshold.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_AUC_THRESHOLD = 0.70  # alert if AUC drops below this
DEFAULT_WINDOW_DAYS = 7       # evaluate weekly


@dataclass
class PerformanceRecord:
    """Single performance measurement."""
    timestamp: datetime
    auc: float
    n_samples: int
    triggered_alert: bool = False


# ─────────────────────────────────────────────────────────────
# Monitor class
# ─────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """Tracks model AUC over sliding windows and raises alerts.

    Usage
    -----
    >>> monitor = PerformanceMonitor(auc_threshold=0.70)
    >>> alert = monitor.log_performance(y_true, y_proba)
    >>> if alert:
    ...     retrain_model()
    """

    def __init__(
        self,
        auc_threshold: float = DEFAULT_AUC_THRESHOLD,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ):
        self.auc_threshold = auc_threshold
        self.window_days = window_days
        self.history: List[PerformanceRecord] = []

    # ─── Public API ─────────────────────────────────────────

    def log_performance(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Compute AUC and check whether retraining is needed.

        Parameters
        ----------
        y_true : array-like
            Ground-truth labels.
        y_proba : array-like
            Predicted probabilities for the positive class.
        timestamp : datetime, optional
            When the batch was scored (defaults to now).

        Returns
        -------
        bool
            True if the AUC has dropped below the threshold (retraining alert).
        """
        ts = timestamp or datetime.now(timezone.utc)
        auc = roc_auc_score(y_true, y_proba)
        alert = auc < self.auc_threshold

        record = PerformanceRecord(
            timestamp=ts,
            auc=round(auc, 6),
            n_samples=len(y_true),
            triggered_alert=alert,
        )
        self.history.append(record)

        if alert:
            logger.warning(
                "⚠️  AUC dropped to %.4f (threshold: %.4f) – retraining recommended!",
                auc, self.auc_threshold,
            )
        else:
            logger.info("AUC = %.4f (threshold: %.4f) – OK", auc, self.auc_threshold)

        return alert

    def get_recent_history(self, days: Optional[int] = None) -> pd.DataFrame:
        """Return performance records within the specified window.

        Parameters
        ----------
        days : int, optional
            Look-back window in days. Defaults to `self.window_days`.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, auc, n_samples, triggered_alert
        """
        if not self.history:
            return pd.DataFrame(columns=["timestamp", "auc", "n_samples", "triggered_alert"])

        days = days or self.window_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [r for r in self.history if r.timestamp >= cutoff]

        return pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "auc": r.auc,
                "n_samples": r.n_samples,
                "triggered_alert": r.triggered_alert,
            }
            for r in recent
        ])

    def compute_weekly_auc(self) -> Optional[float]:
        """Compute the average AUC over the most recent window.

        Returns
        -------
        float or None
            Mean AUC, or None if no records exist.
        """
        df = self.get_recent_history()
        if df.empty:
            return None
        mean_auc = float(df["auc"].mean())
        logger.info("Weekly mean AUC: %.4f (%d batches)", mean_auc, len(df))
        return mean_auc

    def should_retrain(self) -> bool:
        """Check whether the weekly mean AUC is below the threshold.

        Returns
        -------
        bool
            True if retraining is recommended.
        """
        weekly_auc = self.compute_weekly_auc()
        if weekly_auc is None:
            logger.info("No performance records yet – cannot assess retraining need.")
            return False
        needs_retrain = weekly_auc < self.auc_threshold
        if needs_retrain:
            logger.warning(
                "Weekly AUC (%.4f) is below threshold (%.4f) – retraining recommended!",
                weekly_auc, self.auc_threshold,
            )
        return needs_retrain

    def summary(self) -> dict:
        """Return a summary of the monitor state."""
        return {
            "total_records": len(self.history),
            "auc_threshold": self.auc_threshold,
            "window_days": self.window_days,
            "weekly_mean_auc": self.compute_weekly_auc(),
            "retrain_needed": self.should_retrain(),
            "alerts_triggered": sum(1 for r in self.history if r.triggered_alert),
        }

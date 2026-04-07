"""
performance_monitor.py - Model Performance Monitoring

Tracks model performance (AUC) over time and triggers retraining
alerts when the metric drops below a configurable threshold.
History is persisted to a JSON-lines file for durability across restarts.
"""

import json
import logging
import os
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
DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "performance_log.jsonl"
)


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

    History is automatically persisted to a JSON-lines file so it
    survives API restarts.

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
        log_path: Optional[str] = None,
    ):
        self.auc_threshold = auc_threshold
        self.window_days = window_days
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.history: List[PerformanceRecord] = []
        self.load_from_file()

    # ─── Persistence ───────────────────────────────────────

    def _record_to_dict(self, record: PerformanceRecord) -> dict:
        return {
            "timestamp": record.timestamp.isoformat(),
            "auc": record.auc,
            "n_samples": record.n_samples,
            "triggered_alert": record.triggered_alert,
        }

    def _dict_to_record(self, d: dict) -> PerformanceRecord:
        return PerformanceRecord(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            auc=d["auc"],
            n_samples=d["n_samples"],
            triggered_alert=d.get("triggered_alert", False),
        )

    def _ensure_log_dir(self) -> None:
        """Create the parent directory of log_path if it is non-empty."""
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def save_to_file(self) -> None:
        """Write the full history to a JSON-lines file."""
        self._ensure_log_dir()
        with open(self.log_path, "w", encoding="utf-8") as f:
            for record in self.history:
                f.write(json.dumps(self._record_to_dict(record)) + "\n")
        logger.debug("Saved %d records to %s", len(self.history), self.log_path)

    def load_from_file(self) -> None:
        """Load history from a JSON-lines file (if it exists)."""
        if not os.path.exists(self.log_path):
            return
        loaded = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded.append(self._dict_to_record(json.loads(line)))
        self.history = loaded
        logger.info("Loaded %d performance records from %s", len(loaded), self.log_path)

    def _append_to_file(self, record: PerformanceRecord) -> None:
        """Append a single record to the log file."""
        self._ensure_log_dir()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._record_to_dict(record)) + "\n")

    # ─── Public API ─────────────────────────────────────────

    def log_performance(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Compute AUC and check whether retraining is needed.

        Returns True if the AUC has dropped below the threshold.
        Each call is automatically persisted to disk.
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
        self._append_to_file(record)

        if alert:
            logger.warning(
                "⚠️  AUC dropped to %.4f (threshold: %.4f) – retraining recommended!",
                auc, self.auc_threshold,
            )
        else:
            logger.info("AUC = %.4f (threshold: %.4f) – OK", auc, self.auc_threshold)

        return alert

    def get_recent_history(self, days: Optional[int] = None) -> pd.DataFrame:
        """Return performance records within the specified window."""
        if not self.history:
            return pd.DataFrame(
                columns=["timestamp", "auc", "n_samples", "triggered_alert"]
            )
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
        """Compute the average AUC over the most recent window."""
        df = self.get_recent_history()
        if df.empty:
            return None
        mean_auc = float(df["auc"].mean())
        logger.info("Weekly mean AUC: %.4f (%d batches)", mean_auc, len(df))
        return mean_auc

    def should_retrain(self) -> bool:
        """Check whether the weekly mean AUC is below the threshold."""
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
            "log_path": self.log_path,
        }

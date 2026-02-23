"""
drift_detection.py - Feature Drift Detection Module

Implements two statistical approaches to detect data drift between
a reference (training) distribution and a production (serving) distribution:

  1. Kolmogorov–Smirnov (KS) test  — per-feature
  2. Population Stability Index (PSI) — per-feature
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# KS Test
# ─────────────────────────────────────────────────────────────


def ks_test_features(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: List[str] | None = None,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Run a two-sample KS test for each numeric feature.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference (training) data.
    production : pd.DataFrame
        New production data to compare.
    features : list[str] | None
        Specific features to test. If None, all shared numeric columns are used.
    significance : float
        p-value threshold below which drift is flagged.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns: feature, ks_statistic, p_value, drift_detected.
    """
    if features is None:
        features = [
            c for c in reference.select_dtypes(include="number").columns
            if c in production.columns
        ]

    results = []
    for feat in features:
        stat, p_value = stats.ks_2samp(
            reference[feat].dropna(), production[feat].dropna()
        )
        drift = p_value < significance
        results.append({
            "feature": feat,
            "ks_statistic": round(stat, 6),
            "p_value": round(p_value, 6),
            "drift_detected": drift,
        })
        if drift:
            logger.warning("DRIFT detected in '%s' (KS=%.4f, p=%.6f)", feat, stat, p_value)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────


def _compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute PSI for a single feature vector.

    PSI < 0.10  → No significant change
    0.10 ≤ PSI < 0.25 → Moderate drift
    PSI ≥ 0.25 → Significant drift

    Parameters
    ----------
    reference : np.ndarray
        Reference distribution values.
    production : np.ndarray
        Production distribution values.
    n_bins : int
        Number of equal-frequency bins from the reference distribution.

    Returns
    -------
    float
        PSI value.
    """
    # Create bin edges from the reference distribution (equal-frequency)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Count proportions in each bin
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    prod_counts = np.histogram(production, bins=bin_edges)[0]

    ref_pct = ref_counts / len(reference)
    prod_pct = prod_counts / len(production)

    # Avoid log(0) by clipping
    ref_pct = np.clip(ref_pct, 1e-6, None)
    prod_pct = np.clip(prod_pct, 1e-6, None)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


def psi_all_features(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: List[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI for each feature.

    Parameters
    ----------
    reference : pd.DataFrame
        Training data.
    production : pd.DataFrame
        Production data.
    features : list[str] | None
        Columns to evaluate. If None, all shared numeric columns are used.
    n_bins : int
        Number of bins for PSI calculation.

    Returns
    -------
    pd.DataFrame
        Columns: feature, psi, drift_level
    """
    if features is None:
        features = [
            c for c in reference.select_dtypes(include="number").columns
            if c in production.columns
        ]

    results = []
    for feat in features:
        psi = _compute_psi(
            reference[feat].dropna().values,
            production[feat].dropna().values,
            n_bins,
        )
        if psi >= 0.25:
            level = "significant"
        elif psi >= 0.10:
            level = "moderate"
        else:
            level = "none"

        results.append({"feature": feat, "psi": round(psi, 6), "drift_level": level})
        if level != "none":
            logger.warning("PSI drift in '%s': %.4f (%s)", feat, psi, level)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Combined report
# ─────────────────────────────────────────────────────────────

def run_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run both KS and PSI drift checks and return the results.

    Returns
    -------
    ks_results : pd.DataFrame
    psi_results : pd.DataFrame
    """
    logger.info("Running drift detection …")
    ks_results = ks_test_features(reference, production, features)
    psi_results = psi_all_features(reference, production, features)

    n_ks_drift = ks_results["drift_detected"].sum()
    n_psi_drift = (psi_results["drift_level"] != "none").sum()
    logger.info("Drift report: %d features drifted (KS), %d features drifted (PSI)", n_ks_drift, n_psi_drift)

    return ks_results, psi_results

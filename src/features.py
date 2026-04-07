"""
features.py - Feature Engineering Module

Creates domain-specific features for credit default prediction:
  • Average bill amount
  • Credit utilisation ratio
  • Average payment ratio (with fitted cap to avoid data leakage)
  • Delay score
  • Payment trend indicator

Provides both:
  - CreditFeatureTransformer  — sklearn-compatible transformer for use in Pipelines
  - engineer_features()       — convenience function for standalone use
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Column groups
# ─────────────────────────────────────────────────────────────

BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]
PAY_STATUS_COLS = [f"pay_{i}" for i in range(1, 7)]


# ─────────────────────────────────────────────────────────────
# Sklearn-compatible Feature Transformer
# ─────────────────────────────────────────────────────────────


class CreditFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that engineers credit-risk features.

    During `fit()`, learns the 99th-percentile cap for `avg_payment_ratio`
    from training data only. During `transform()`, applies all feature
    engineering steps using the stored cap — preventing data leakage.

    Parameters
    ----------
    payment_ratio_quantile : float
        Quantile used to cap ``avg_payment_ratio`` (default 0.99).
    """

    def __init__(self, payment_ratio_quantile: float = 0.99):
        self.payment_ratio_quantile = payment_ratio_quantile

    def fit(self, X: pd.DataFrame, y=None):
        """Learn the avg_payment_ratio cap from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (must contain bill_amt and pay_amt columns).
        y : ignored
        """
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        avg_pay = df[PAY_AMT_COLS].mean(axis=1)
        avg_bill = df[BILL_COLS].mean(axis=1)
        ratio = avg_pay / avg_bill.replace(0, np.nan)
        ratio = ratio.fillna(0)

        self.payment_ratio_cap_ = float(ratio.quantile(self.payment_ratio_quantile))
        logger.info(
            "Fitted payment_ratio_cap = %.4f (q=%.2f)",
            self.payment_ratio_cap_, self.payment_ratio_quantile,
        )
        return self

    # Documented ranges from the UCI Credit Card Default paper
    _CATEGORICAL_RANGES = {
        "sex": (1, 2),           # 1=male, 2=female
        "education": (1, 4),     # 1=grad school, 2=university, 3=high school, 4=others
        "marriage": (1, 3),      # 1=married, 2=single, 3=others
    }

    def _validate_categoricals(self, df: pd.DataFrame) -> None:
        """Warn about undocumented categorical values and clip to known ranges.

        The UCI dataset contains undocumented values (e.g., education=0, 5, 6).
        These are clipped to the nearest documented boundary so the model
        receives inputs within a known distribution.
        """
        import warnings
        for col, (lo, hi) in self._CATEGORICAL_RANGES.items():
            if col not in df.columns:
                continue
            out_of_range = ~df[col].between(lo, hi)
            if out_of_range.any():
                bad_vals = sorted(df.loc[out_of_range, col].unique())
                warnings.warn(
                    f"Column '{col}' has undocumented values {bad_vals} "
                    f"(expected {lo}–{hi}). Clipping to nearest boundary.",
                    UserWarning,
                    stacklevel=3,
                )
                df[col] = df[col].clip(lower=lo, upper=hi)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Raw features (23 columns).

        Returns
        -------
        pd.DataFrame
            DataFrame with 5 additional engineered features (28 columns).
        """
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Validate categorical inputs (UCI dataset documented ranges)
        self._validate_categoricals(df)

        # 1. Average bill amount
        df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)

        # 2. Credit utilisation ratio
        avg_bill = df[BILL_COLS].mean(axis=1)
        df["credit_utilization"] = avg_bill / df["limit_bal"].replace(0, np.nan)
        df["credit_utilization"] = df["credit_utilization"].fillna(0)

        # 3. Average payment ratio (capped with the FITTED value)
        avg_pay = df[PAY_AMT_COLS].mean(axis=1)
        df["avg_payment_ratio"] = avg_pay / avg_bill.replace(0, np.nan)
        df["avg_payment_ratio"] = df["avg_payment_ratio"].fillna(0)
        df["avg_payment_ratio"] = df["avg_payment_ratio"].clip(
            upper=self.payment_ratio_cap_
        )

        # 4. Delay score
        delay_mask = df[PAY_STATUS_COLS].apply(lambda col: col > 0)
        df["delay_score"] = delay_mask.sum(axis=1)

        # 5. Payment trend (slope)
        x = np.arange(1, 7)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        pay_values = df[PAY_AMT_COLS].values
        y_mean = pay_values.mean(axis=1, keepdims=True)
        slopes = ((pay_values - y_mean) * (x - x_mean)).sum(axis=1) / x_var
        df["payment_trend"] = slopes

        return df

    def get_feature_names_out(self, input_features=None):
        """Return output feature names (required for newer sklearn versions)."""
        base = list(input_features) if input_features is not None else []
        return base + [
            "avg_bill_amt", "credit_utilization", "avg_payment_ratio",
            "delay_score", "payment_trend",
        ]


# ─────────────────────────────────────────────────────────────
# Convenience wrapper — EDA / notebook use only
# ─────────────────────────────────────────────────────────────


def engineer_features(
    df: pd.DataFrame,
    payment_ratio_cap: Optional[float] = None,
) -> pd.DataFrame:
    """Apply all feature engineering steps (standalone, outside a Pipeline).

    .. warning::
       This function fits the payment-ratio cap on the **provided** DataFrame.
       If called on test / production data, the cap will come from that data
       (data leakage). For training + serving, use ``CreditFeatureTransformer``
       inside a ``sklearn.Pipeline`` instead.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned credit-card dataset.
    payment_ratio_cap : float, optional
        Pre-computed cap for avg_payment_ratio. If None, the 99th percentile
        of the current df is used.

    Returns
    -------
    pd.DataFrame
        Dataframe with 5 additional engineered features.
    """
    import warnings
    warnings.warn(
        "engineer_features() fits the payment-ratio cap on the provided df. "
        "For training/serving, use CreditFeatureTransformer in a Pipeline.",
        UserWarning,
        stacklevel=2,
    )
    logger.info("Engineering features (standalone) …")
    transformer = CreditFeatureTransformer()
    transformer.fit(df)
    if payment_ratio_cap is not None:
        transformer.payment_ratio_cap_ = payment_ratio_cap
    result = transformer.transform(df)
    logger.info("Feature engineering complete – new shape: %s", result.shape)
    return result


# Explicit alias for EDA use
engineer_features_for_eda = engineer_features


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_cleaned_data

    df = load_cleaned_data()
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", UserWarning)
        df = engineer_features(df)
    print(df[["avg_bill_amt", "credit_utilization", "avg_payment_ratio",
              "delay_score", "payment_trend"]].describe())

"""
features.py - Feature Engineering Module

Creates domain-specific features for credit default prediction:
  • Average bill amount
  • Credit utilisation ratio
  • Average payment ratio
  • Delay score
  • Payment trend indicator
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Column groups
# ─────────────────────────────────────────────────────────────

BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]
PAY_STATUS_COLS = [f"pay_{i}" for i in range(1, 7)]


# ─────────────────────────────────────────────────────────────
# Individual feature functions
# ─────────────────────────────────────────────────────────────


def add_avg_bill_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Average of billing amounts across 6 months.

    A high average bill indicates larger outstanding balances,
    which can signal higher default risk.
    """
    df = df.copy()
    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    return df


def add_credit_utilization_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Mean(bill_amt) / limit_bal — how much of the credit line is used.

    Values > 1.0 indicate the customer has exceeded their credit limit.
    """
    df = df.copy()
    avg_bill = df[BILL_COLS].mean(axis=1)
    df["credit_utilization"] = avg_bill / df["limit_bal"].replace(0, np.nan)
    df["credit_utilization"] = df["credit_utilization"].fillna(0)
    return df


def add_avg_payment_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Mean(pay_amt) / Mean(bill_amt) — how much of the bill is paid on average.

    A ratio close to 1 or above means the customer pays off most of their
    balance; values near 0 indicate minimum or no payments.
    """
    df = df.copy()
    avg_pay = df[PAY_AMT_COLS].mean(axis=1)
    avg_bill = df[BILL_COLS].mean(axis=1)
    df["avg_payment_ratio"] = avg_pay / avg_bill.replace(0, np.nan)
    df["avg_payment_ratio"] = df["avg_payment_ratio"].fillna(0)
    # Cap extreme values at the 99th percentile
    cap = df["avg_payment_ratio"].quantile(0.99)
    df["avg_payment_ratio"] = df["avg_payment_ratio"].clip(upper=cap)
    return df


def add_delay_score(df: pd.DataFrame) -> pd.DataFrame:
    """Count of months where repayment status indicates a delay (> 0).

    PAY_x values:
      -2 = no consumption, -1 = paid in full, 0 = revolving credit,
       1 = 1 month delay, 2 = 2 months delay, … , 8 = 8 months delay, 9 = ≥9 months
    A higher delay score reflects worse payment behaviour.
    """
    df = df.copy()
    delay_mask = df[PAY_STATUS_COLS].apply(lambda col: col > 0)
    df["delay_score"] = delay_mask.sum(axis=1)
    return df


def add_payment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Trend in payment amounts over the 6-month window.

    Computed as the slope of a simple linear regression of pay_amt values
    over time indices 1–6.  A positive trend means the customer is paying
    more over time (improving); a negative trend signals deterioration.
    """
    df = df.copy()
    x = np.arange(1, 7)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    pay_values = df[PAY_AMT_COLS].values  # (n_samples, 6)
    y_mean = pay_values.mean(axis=1, keepdims=True)
    slopes = ((pay_values - y_mean) * (x - x_mean)).sum(axis=1) / x_var
    df["payment_trend"] = slopes
    return df


# ─────────────────────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned credit-card dataset (output of data_loader.clean_dataframe).

    Returns
    -------
    pd.DataFrame
        Dataframe with 5 additional engineered features.
    """
    logger.info("Engineering features …")
    df = add_avg_bill_amount(df)
    df = add_credit_utilization_ratio(df)
    df = add_avg_payment_ratio(df)
    df = add_delay_score(df)
    df = add_payment_trend(df)
    logger.info("Feature engineering complete – new shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_cleaned_data

    df = load_cleaned_data()
    df = engineer_features(df)
    print(df[["avg_bill_amt", "credit_utilization", "avg_payment_ratio",
              "delay_score", "payment_trend"]].describe())

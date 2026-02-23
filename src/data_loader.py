"""
data_loader.py - Data Loading and Preprocessing Module

Downloads the UCI Credit Card Default dataset, cleans column names,
and provides reusable loading functions for the ML pipeline.
"""

import os
import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_FILENAME = "raw_credit_card_default.xls"
CLEANED_FILENAME = "credit_card_default_cleaned.csv"

# Standardised column name mapping
COLUMN_RENAME_MAP = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_1",   # original data labels first month as PAY_0
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "default",
}


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def download_dataset(data_dir: str = DEFAULT_DATA_DIR, force: bool = False) -> str:
    """Download the raw dataset from the UCI repository.

    Parameters
    ----------
    data_dir : str
        Directory to save the downloaded file.
    force : bool
        If True, re-download even if the file already exists.

    Returns
    -------
    str
        Absolute path to the downloaded file.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, RAW_FILENAME)

    if os.path.exists(filepath) and not force:
        logger.info("Raw dataset already exists at %s", filepath)
        return filepath

    logger.info("Downloading dataset from UCI repository …")
    response = requests.get(UCI_URL, timeout=120)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    logger.info("Dataset saved to %s", filepath)
    return filepath


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the raw dataframe.

    Steps:
        1. Rename columns to snake_case.
        2. Drop the ID column (not a feature).
        3. Fix undocumented category codes in EDUCATION and MARRIAGE.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from the XLS file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for EDA and feature engineering.
    """
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Drop ID if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Education: 0, 5, 6 are undocumented → group into "other" (4)
    df["education"] = df["education"].apply(lambda x: 4 if x not in [1, 2, 3, 4] else x)

    # Marriage: 0 is undocumented → group into "other" (3)
    df["marriage"] = df["marriage"].apply(lambda x: 3 if x not in [1, 2, 3] else x)

    return df


def load_raw_data(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the raw XLS file from disk (downloads if necessary).

    Parameters
    ----------
    data_dir : str
        Directory where the raw file is stored.

    Returns
    -------
    pd.DataFrame
        Raw dataframe (header row = row 1 in the XLS).
    """
    filepath = download_dataset(data_dir)
    df = pd.read_excel(filepath, header=1, engine="xlrd")
    logger.info("Loaded raw data with shape %s", df.shape)
    return df


def load_cleaned_data(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the cleaned CSV.  If it doesn't exist, create it first.

    Parameters
    ----------
    data_dir : str
        Directory where the cleaned CSV is stored.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    csv_path = os.path.join(data_dir, CLEANED_FILENAME)

    if os.path.exists(csv_path):
        logger.info("Loading cleaned data from %s", csv_path)
        return pd.read_csv(csv_path)

    # Build cleaned CSV from raw download
    logger.info("Cleaned CSV not found – building from raw data …")
    df_raw = load_raw_data(data_dir)
    df_clean = clean_dataframe(df_raw)
    save_cleaned_data(df_clean, data_dir)
    return df_clean


def save_cleaned_data(df: pd.DataFrame, data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Persist the cleaned dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    data_dir : str
        Target directory.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, CLEANED_FILENAME)
    df.to_csv(csv_path, index=False)
    logger.info("Cleaned data saved to %s", csv_path)
    return csv_path


# ─────────────────────────────────────────────────────────────
# Convenience CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_cleaned_data()
    print(f"Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())

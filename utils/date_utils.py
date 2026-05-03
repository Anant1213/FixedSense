"""Date utility functions for FixedSense.

Handles consistent date management across the system, especially when using sample data.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config.settings import settings


def get_sample_data_date_range() -> tuple[date, date]:
    """
    Get the actual date range available in sample data.

    Returns:
        (min_date, max_date) tuple from sample data CSV
    """
    sample_file = settings.SAMPLE_DATA_PATH / "us_treasury_yields_2022_2024.csv"

    if not sample_file.exists():
        raise FileNotFoundError(f"Sample data not found: {sample_file}")

    df = pd.read_csv(sample_file)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    min_date = df["date"].min()
    max_date = df["date"].max()

    return min_date, max_date


def get_as_of_date() -> date:
    """
    Get the as_of_date to use in the system.

    When USE_SAMPLE_DATA=true, returns the latest date in sample data.
    Otherwise, returns today's date.

    Returns:
        date object to use as system reference date
    """
    if settings.USE_SAMPLE_DATA:
        _, max_date = get_sample_data_date_range()
        return max_date
    else:
        return date.today()


def get_data_date_range() -> tuple[date, date]:
    """
    Get the date range for data loading.

    When USE_SAMPLE_DATA=true, returns the full range of sample data.
    Otherwise, returns last 504 days (2 years) from today.

    Returns:
        (start_date, end_date) tuple
    """
    if settings.USE_SAMPLE_DATA:
        return get_sample_data_date_range()
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=504)
        return start_date, end_date

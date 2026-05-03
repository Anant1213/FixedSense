"""Data cleaning: forward-fill missing dates, outlier removal, alignment."""

import logging

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and transforms raw data for analysis."""

    @staticmethod
    def align_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
        """
        Align data to business day calendar.

        Removes weekends/holidays where applicable.
        Forward-fills missing business days (e.g., after a holiday).

        Args:
            df: DataFrame with 'date' column

        Returns:
            DataFrame aligned to business days
        """
        df = df.copy()

        if "date" not in df.columns:
            return df

        df["date"] = pd.to_datetime(df["date"])

        # Get business days in range
        min_date = df["date"].min()
        max_date = df["date"].max()
        business_days = pd.bdate_range(start=min_date, end=max_date)

        # Forward-fill to business days
        df = df.set_index("date").reindex(business_days)
        df = df.fillna(method="ffill")
        df = df.reset_index()
        df = df.rename(columns={"index": "date"})

        logger.info(f"Aligned {len(df)} rows to business day calendar")
        return df

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        column: str,
        std_threshold: float | None = None,
        method: str = "mark",
    ) -> pd.DataFrame:
        """
        Detect and handle outliers.

        Args:
            df: Input DataFrame
            column: Column to check for outliers
            std_threshold: Number of standard deviations. Defaults to settings.OUTLIER_STD_THRESHOLD
            method: "remove" (drop rows) or "mark" (add flag column)

        Returns:
            DataFrame with outliers removed or marked
        """
        df = df.copy()

        if std_threshold is None:
            std_threshold = settings.OUTLIER_STD_THRESHOLD

        if column not in df.columns:
            return df

        # Rolling mean and std
        rolling_mean = df[column].rolling(window=21, center=True).mean()
        rolling_std = df[column].rolling(window=21, center=True).std()

        # Z-scores
        z_scores = np.abs((df[column] - rolling_mean) / (rolling_std + 1e-6))
        is_outlier = z_scores > std_threshold

        num_outliers = is_outlier.sum()

        if method == "mark":
            df["is_outlier"] = is_outlier
            logger.info(f"Marked {num_outliers} outliers in {column}")
        elif method == "remove":
            df = df[~is_outlier]
            logger.info(f"Removed {num_outliers} outliers from {column}")

        return df

    @staticmethod
    def interpolate_missing_values(
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: str = "linear",
    ) -> pd.DataFrame:
        """
        Interpolate missing values.

        Args:
            df: Input DataFrame
            columns: Columns to interpolate. Defaults to all numeric columns
            method: "linear", "ffill", or "bfill"

        Returns:
            DataFrame with interpolated values
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                if method == "linear":
                    df[col] = df[col].interpolate(method="linear")
                elif method == "ffill":
                    df[col] = df[col].fillna(method="ffill")
                elif method == "bfill":
                    df[col] = df[col].fillna(method="bfill")

        logger.info(f"Interpolated {len(columns)} columns using {method}")
        return df

    @staticmethod
    def compute_daily_changes(
        df: pd.DataFrame,
        columns: list[str] | None = None,
        group_by: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute daily changes (diffs) and returns.

        Args:
            df: Input DataFrame
            columns: Columns to compute changes for. Defaults to all numeric
            group_by: Column to group by before computing diffs (e.g., 'tenor' for tenor-wise changes)

        Returns:
            DataFrame with added columns: {col}_change and {col}_pct_change
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if group_by:
            for col in columns:
                df[f"{col}_change"] = df.groupby(group_by)[col].diff()
                df[f"{col}_pct_change"] = df.groupby(group_by)[col].pct_change()
        else:
            for col in columns:
                df[f"{col}_change"] = df[col].diff()
                df[f"{col}_pct_change"] = df[col].pct_change()

        logger.info(f"Computed daily changes for {len(columns)} columns")
        return df

    @staticmethod
    def compute_rolling_statistics(
        df: pd.DataFrame,
        columns: list[str] | None = None,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Compute rolling mean and volatility.

        Args:
            df: Input DataFrame
            columns: Columns to compute statistics for
            window: Rolling window size (default 21 = 1 month of trading days)

        Returns:
            DataFrame with added columns: {col}_mean_{window}, {col}_vol_{window}
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                df[f"{col}_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_vol_{window}"] = df[col].rolling(window).std()

        logger.info(f"Computed {window}-day rolling statistics for {len(columns)} columns")
        return df

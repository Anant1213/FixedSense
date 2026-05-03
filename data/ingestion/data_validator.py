"""Data validation module for incoming market data."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    warnings: list[str]
    errors: list[str]

    def __str__(self) -> str:
        """Pretty print validation result."""
        lines = [f"Valid: {self.is_valid}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


class DataValidator:
    """Validates incoming yield curve and price data."""

    @staticmethod
    def validate_yield_data(df: pd.DataFrame) -> ValidationResult:
        """
        Validate treasury yield data.

        Checks:
        1. Schema: required columns present
        2. Nulls: not more than MAX_NULL_PCT per column
        3. Range: yields between YIELD_MIN_PCT and YIELD_MAX_PCT
        4. Staleness: most recent date is recent
        5. Monotonicity: yield curve shape is reasonable (warn on inversions)
        6. Duplicates: no duplicate date+tenor combinations

        Args:
            df: DataFrame with columns [date, tenor, yield_pct]

        Returns:
            ValidationResult with is_valid, warnings, errors
        """
        errors = []
        warnings = []

        # 1. SCHEMA CHECK
        required_cols = {"date", "tenor", "yield_pct"}
        if not required_cols.issubset(df.columns):
            errors.append(f"Missing columns. Required: {required_cols}, got: {set(df.columns)}")
            return ValidationResult(is_valid=False, warnings=warnings, errors=errors)

        # 2. NULL CHECK
        for col in ["yield_pct"]:
            null_pct = df[col].isna().sum() / len(df)
            if null_pct > settings.MAX_NULL_PCT:
                errors.append(f"Column '{col}' has {null_pct:.1%} nulls (max allowed: {settings.MAX_NULL_PCT:.1%})")
            elif null_pct > 0:
                warnings.append(f"Column '{col}' has {null_pct:.1%} nulls")

        # 3. RANGE CHECK (yields should be reasonable)
        min_yield = df["yield_pct"].min()
        max_yield = df["yield_pct"].max()

        if min_yield < settings.YIELD_MIN_PCT:
            errors.append(f"Yield below minimum: {min_yield:.2%} (min allowed: {settings.YIELD_MIN_PCT:.2%})")
        if max_yield > settings.YIELD_MAX_PCT:
            errors.append(f"Yield above maximum: {max_yield:.2%} (max allowed: {settings.YIELD_MAX_PCT:.2%})")

        # 4. STALENESS CHECK
        if "date" in df.columns and len(df) > 0:
            most_recent = df["date"].max()
            age_days = (pd.Timestamp.now().date() - most_recent).days

            if age_days > settings.DATA_STALENESS_DAYS:
                errors.append(f"Data is stale: {age_days} days old (max allowed: {settings.DATA_STALENESS_DAYS})")

        # 5. MONOTONICITY CHECK (curve should generally be upward sloping, warn on inversions)
        if "tenor" in df.columns and "yield_pct" in df.columns:
            # Check most recent date
            most_recent_date = df["date"].max()
            latest_curve = df[df["date"] == most_recent_date].sort_values("tenor")

            if len(latest_curve) > 2:
                # Check for yield inversions (longer tenor with lower yield)
                inversions = 0
                for i in range(len(latest_curve) - 1):
                    if latest_curve.iloc[i]["yield_pct"] > latest_curve.iloc[i+1]["yield_pct"]:
                        inversions += 1

                if inversions > len(latest_curve) / 3:  # More than 1/3 are inverted
                    warnings.append(f"Yield curve has {inversions} inversions on {most_recent_date}")

        # 6. DUPLICATE CHECK
        if "date" in df.columns and "tenor" in df.columns:
            duplicates = df.groupby(["date", "tenor"]).size()
            if (duplicates > 1).any():
                num_dups = (duplicates > 1).sum()
                errors.append(f"Found {num_dups} date+tenor duplicates")

        # 7. OUTLIER CHECK (values > 4 std devs from rolling mean)
        if "yield_pct" in df.columns and len(df) > 10:
            df_sorted = df.sort_values("date")
            rolling_mean = df_sorted["yield_pct"].rolling(window=21, center=True).mean()
            rolling_std = df_sorted["yield_pct"].rolling(window=21, center=True).std()

            z_scores = np.abs((df_sorted["yield_pct"] - rolling_mean) / (rolling_std + 1e-6))
            outliers = (z_scores > settings.OUTLIER_STD_THRESHOLD).sum()

            if outliers > 0:
                warnings.append(f"Found {outliers} potential outliers (> {settings.OUTLIER_STD_THRESHOLD} std devs)")

        # Determine if data is valid (errors = invalid, warnings = valid but concerning)
        is_valid = len(errors) == 0

        return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)

    @staticmethod
    def validate_bond_prices(df: pd.DataFrame) -> ValidationResult:
        """
        Validate bond price data (from Yahoo Finance or similar).

        Checks:
        1. Schema: ticker, date, close, volume
        2. Prices: positive, reasonable range (prices 50-150 for ETFs)
        3. Volumes: non-negative

        Args:
            df: DataFrame with bond ETF price data

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # SCHEMA
        required_cols = {"date", "ticker", "close", "volume"}
        if not required_cols.issubset(df.columns):
            errors.append(f"Missing columns: {required_cols - set(df.columns)}")
            return ValidationResult(is_valid=False, warnings=warnings, errors=errors)

        # PRICES
        if (df["close"] <= 0).any():
            errors.append("Negative or zero prices found")

        if (df["close"] < 30).any() or (df["close"] > 300).any():
            warnings.append("Some prices outside typical range (30-300) — check for data errors")

        # VOLUMES
        if (df["volume"] < 0).any():
            errors.append("Negative volumes found")

        is_valid = len(errors) == 0

        return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)

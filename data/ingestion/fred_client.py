"""FRED API client for fetching US Treasury yields + fallback to sample data."""

import logging
from datetime import date, datetime, timedelta

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class FREDClient:
    """
    Fetch US Treasury yields from FRED API or use sample data.

    FRED docs: https://fred.stlouisfed.org/docs/api/
    """

    def __init__(self, use_sample_data: bool = True):
        """
        Initialize FRED client.

        Args:
            use_sample_data: If True, load from sample_data/ CSVs instead of API
        """
        self.use_sample_data = use_sample_data or not settings.FRED_API_KEY

        if not self.use_sample_data:
            try:
                import fredapi
                self.fred = fredapi.Fred(api_key=settings.FRED_API_KEY)
                logger.info("FRED client initialized with API key")
            except ImportError:
                logger.warning("fredapi not installed. Falling back to sample data.")
                self.use_sample_data = True
            except Exception as e:
                logger.warning(f"FRED initialization failed: {e}. Using sample data.")
                self.use_sample_data = True

    def fetch_treasury_yields(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch US Treasury yields for all configured tenors.

        Args:
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with columns: [date, tenor, yield_pct]
            tenor: float in years (0.083 for 1MO, 30.0 for 30Y)
            yield_pct: float (0.0725 for 7.25%)
        """
        if self.use_sample_data:
            return self._fetch_sample_data(start_date, end_date)

        # Fetch from FRED API
        dfs = []
        for tenor, series_id in settings.tenor_to_series.items():
            try:
                df = self._fetch_single_series(series_id, start_date, end_date)
                df["tenor"] = tenor
                dfs.append(df)
                logger.info(f"Fetched {series_id} ({tenor}Y): {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to fetch {series_id}: {e}")

        if not dfs:
            logger.warning("Failed to fetch any FRED data. Falling back to sample data.")
            return self._fetch_sample_data(start_date, end_date)

        result = pd.concat(dfs, ignore_index=True)
        result = result.rename(columns={series_id: "yield_pct"})
        result = result[["date", "tenor", "yield_pct"]].sort_values("date")

        # Forward-fill missing values (FRED has gaps on holidays)
        result["yield_pct"] = result.groupby("tenor")["yield_pct"].fillna(method="ffill")

        logger.info(f"Fetched {len(result)} total yield observations")
        return result

    def _fetch_single_series(
        self,
        series_id: str,
        start_date: date,
        end_date: date,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series with retry logic.

        Args:
            series_id: FRED series ID (e.g., 'DGS10')
            start_date: Start date
            end_date: End date
            max_retries: Maximum retry attempts

        Returns:
            DataFrame with columns: [date, series_id]
        """
        import time
        import fredapi

        for attempt in range(max_retries):
            try:
                # FRED rate limit: 120 requests/minute
                time.sleep(0.5)

                series = self.fred.get_series(series_id, start_date, end_date)

                df = pd.DataFrame({
                    "date": series.index.date,
                    series_id: series.values,
                })

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"FRED fetch failed (attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def _fetch_sample_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Load sample data from CSV instead of API.

        Uses sample_data/us_treasury_yields_2022_2024.csv
        Filters to requested date range.
        """
        sample_file = settings.SAMPLE_DATA_PATH / "us_treasury_yields_2022_2024.csv"

        if not sample_file.exists():
            logger.error(f"Sample data file not found: {sample_file}")
            return pd.DataFrame()

        df = pd.read_csv(sample_file)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Convert columns (DGS1MO, DGS3MO, ...) to (tenor, yield_pct) format
        tenor_series_map = {v: k for k, v in settings.tenor_to_series.items()}

        data = []
        for col_name, tenor in tenor_series_map.items():
            if col_name in df.columns:
                subset = df[["date", col_name]].copy()
                subset["tenor"] = tenor
                subset = subset.rename(columns={col_name: "yield_pct"})
                data.append(subset[["date", "tenor", "yield_pct"]])

        if not data:
            logger.error(f"No yield columns found in {sample_file}")
            return pd.DataFrame()

        result = pd.concat(data, ignore_index=True)

        # Filter to date range
        result = result[(result["date"] >= start_date) & (result["date"] <= end_date)]
        result = result.sort_values("date")

        logger.info(f"Loaded {len(result)} sample data observations from {sample_file}")
        return result


# Global FRED client instance
def get_fred_client(use_sample_data: bool | None = None) -> FREDClient:
    """Get FRED client with sample data or API."""
    if use_sample_data is None:
        use_sample_data = settings.USE_SAMPLE_DATA
    return FREDClient(use_sample_data=use_sample_data)

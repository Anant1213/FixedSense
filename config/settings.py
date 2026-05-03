"""
Central configuration for FixedSense.
All magic numbers, thresholds, and settings live here.
Loaded from .env file with Pydantic validation.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class StorageMode(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class DayCountConvention(str, Enum):
    ACT_365 = "ACT/365"  # Indian bond market standard
    ACT_ACT = "ACT/ACT"  # US Treasuries
    THIRTY_360 = "30/360"


class RiskFreeRateSeries(str, Enum):
    """Economic series to use as risk-free rate proxy"""
    DGS10 = "DGS10"  # 10-year US Treasury (common proxy)
    DGS2 = "DGS2"    # 2-year Treasury


class Settings(BaseSettings):
    """
    Production-grade configuration.

    Environment variables override defaults.
    Type validation enforced by Pydantic.
    """

    # === PROJECT PATHS ===
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    SAMPLE_DATA_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "sample_data"
    )

    # === FRED API ===
    FRED_API_KEY: str = Field(default="", description="FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html")
    USE_SAMPLE_DATA: bool = Field(default=True, description="Use sample data instead of live FRED API")

    # === STORAGE ===
    STORAGE_MODE: StorageMode = Field(default=StorageMode.LOCAL)
    S3_BUCKET: str = Field(default="fixedsense-data")
    AWS_REGION: str = Field(default="us-east-1")
    AWS_ACCESS_KEY_ID: str = Field(default="")
    AWS_SECRET_ACCESS_KEY: str = Field(default="")

    # === S3 ZONES ===
    S3_RAW_ZONE: str = Field(default="raw")
    S3_PROCESSED_ZONE: str = Field(default="processed")
    S3_ANALYTICS_ZONE: str = Field(default="analytics")

    # === PORTFOLIO & CONFIG ===
    PORTFOLIO_CONFIG_PATH: Path = Field(default="config/portfolio.yaml")
    BENCHMARK_CONFIG_PATH: Path = Field(default="config/benchmark.yaml")
    SCENARIOS_CONFIG_PATH: Path = Field(default="config/scenarios.yaml")

    # === YIELD CURVE ===
    DAY_COUNT_CONVENTION: DayCountConvention = Field(default=DayCountConvention.ACT_365)
    RISK_FREE_RATE_SERIES: RiskFreeRateSeries = Field(default=RiskFreeRateSeries.DGS10)

    # FRED series for treasury yields (tenors in years)
    FRED_SERIES_DGS1MO: str = Field(default="DGS1MO")  # 1-month
    FRED_SERIES_DGS3MO: str = Field(default="DGS3MO")  # 3-month
    FRED_SERIES_DGS6MO: str = Field(default="DGS6MO")  # 6-month
    FRED_SERIES_DGS1: str = Field(default="DGS1")      # 1-year
    FRED_SERIES_DGS2: str = Field(default="DGS2")      # 2-year
    FRED_SERIES_DGS3: str = Field(default="DGS3")      # 3-year
    FRED_SERIES_DGS5: str = Field(default="DGS5")      # 5-year
    FRED_SERIES_DGS7: str = Field(default="DGS7")      # 7-year
    FRED_SERIES_DGS10: str = Field(default="DGS10")    # 10-year
    FRED_SERIES_DGS20: str = Field(default="DGS20")    # 20-year
    FRED_SERIES_DGS30: str = Field(default="DGS30")    # 30-year

    # Key rate DV01 tenor buckets (years)
    KEY_RATE_TENORS: list[float] = Field(
        default=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        description="Tenor points for KR01 computation"
    )

    # === PCA MODEL ===
    PCA_LOOKBACK_DAYS: int = Field(default=504, description="~2 years of trading days")
    PCA_NUM_FACTORS: int = Field(default=3, description="level, slope, curvature")
    PCA_REFIT_FREQUENCY_DAYS: int = Field(default=21, description="Refit PCA model weekly")

    # === MONTE CARLO ===
    MONTE_CARLO_PATHS: int = Field(default=10000, description="Number of simulation scenarios")
    MONTE_CARLO_HORIZON_DAYS: int = Field(default=1, description="1-day VaR")
    MONTE_CARLO_SEED: int | None = Field(default=None, description="Random seed for reproducibility")

    # === VaR / CVaR ===
    VAR_CONFIDENCE: float = Field(default=0.95, description="95% confidence level")
    ES_CONFIDENCE: float = Field(default=0.975, description="FRTB 97.5% confidence level")

    # === CREDIT RISK ===
    CREDIT_SPREAD_VOL_BPS: float = Field(default=30, description="Credit spread volatility")
    SPREAD_VOL_LOOKBACK_DAYS: int = Field(default=252, description="1 year for spread vol estimation")

    # === ALERTS ===
    ALERT_VAR_THRESHOLD_PCT: float = Field(
        default=0.80,
        description="Alert if VaR utilization > 80% of limit"
    )
    ALERT_CONCENTRATION_THRESHOLD_PCT: float = Field(
        default=0.60,
        description="Alert if any tenor > 60% of total KR01"
    )
    ALERT_SPREAD_WIDENING_BPS: float = Field(
        default=20,
        description="Alert if any bond's spread widens > 20bps in one day"
    )
    ALERT_PNL_DEVIATION_MULTIPLIER: float = Field(
        default=3.0,
        description="Alert if |actual_pnl - predicted| > 3x typical residual"
    )

    # === KAFKA (optional) ===
    KAFKA_BROKER: str = Field(default="localhost:9092")
    KAFKA_TOPIC_RISK: str = Field(default="fixedsense.risk.events")
    KAFKA_TOPIC_ALERTS: str = Field(default="fixedsense.alerts")
    KAFKA_ENABLED: bool = Field(default=False, description="Enable Kafka streaming")

    # === LOGGING ===
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # === DATA VALIDATION ===
    MAX_NULL_PCT: float = Field(default=0.05, description="Reject if > 5% nulls per column")
    YIELD_MIN_PCT: float = Field(default=-2.0, description="Minimum reasonable yield")
    YIELD_MAX_PCT: float = Field(default=25.0, description="Maximum reasonable yield")
    DATA_STALENESS_DAYS: int = Field(default=2, description="Alert if data older than 2 biz days")
    OUTLIER_STD_THRESHOLD: float = Field(
        default=4.0,
        description="Mark as outlier if > 4 std devs from rolling mean"
    )

    # === REPORTING ===
    BACKTEST_WINDOW_DAYS: int = Field(default=250, description="~1 year of trading days")
    BASEL_YELLOW_EXCEPTIONS: int = Field(default=5, description="Basel traffic light: start yellow at 5")
    BASEL_RED_EXCEPTIONS: int = Field(default=10, description="Basel traffic light: red at 10")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "allow",
    }

    @property
    def all_fred_series(self) -> list[str]:
        """Return all FRED series IDs as list"""
        return [
            self.FRED_SERIES_DGS1MO,
            self.FRED_SERIES_DGS3MO,
            self.FRED_SERIES_DGS6MO,
            self.FRED_SERIES_DGS1,
            self.FRED_SERIES_DGS2,
            self.FRED_SERIES_DGS3,
            self.FRED_SERIES_DGS5,
            self.FRED_SERIES_DGS7,
            self.FRED_SERIES_DGS10,
            self.FRED_SERIES_DGS20,
            self.FRED_SERIES_DGS30,
        ]

    @property
    def tenor_to_series(self) -> dict[float, str]:
        """Map tenor years to FRED series ID"""
        return {
            1/12: self.FRED_SERIES_DGS1MO,
            3/12: self.FRED_SERIES_DGS3MO,
            6/12: self.FRED_SERIES_DGS6MO,
            1.0: self.FRED_SERIES_DGS1,
            2.0: self.FRED_SERIES_DGS2,
            3.0: self.FRED_SERIES_DGS3,
            5.0: self.FRED_SERIES_DGS5,
            7.0: self.FRED_SERIES_DGS7,
            10.0: self.FRED_SERIES_DGS10,
            20.0: self.FRED_SERIES_DGS20,
            30.0: self.FRED_SERIES_DGS30,
        }


# Global settings instance
settings = Settings()

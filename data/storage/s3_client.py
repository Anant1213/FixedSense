"""
Storage abstraction layer with pluggable backends.

StorageBackend ABC defines the interface for all storage operations.
Implementations: LocalStorage (filesystem), S3Storage (AWS S3).
All subsequent layers use get_storage() factory — never import storage directly.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from config.settings import StorageMode, settings

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for all storage implementations."""

    @abstractmethod
    def write_parquet(
        self,
        df: pd.DataFrame,
        zone: str,
        data_type: str,
        date: date,
    ) -> str:
        """
        Write DataFrame to Parquet with date partitioning.

        Args:
            df: DataFrame to write
            zone: Storage zone (raw, processed, analytics)
            data_type: Type of data (e.g., 'treasury_yields', 'pca_model')
            date: Partition date (YYYY/MM/DD)

        Returns:
            Path where file was written
        """

    @abstractmethod
    def read_parquet(
        self,
        zone: str,
        data_type: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Read Parquet files within date range.

        Args:
            zone: Storage zone
            data_type: Type of data
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            Concatenated DataFrame from all matching partitions
        """

    @abstractmethod
    def list_available_dates(
        self,
        zone: str,
        data_type: str,
    ) -> list[date]:
        """
        List all available dates for a given zone/type.

        Returns:
            Sorted list of dates with available data
        """

    @abstractmethod
    def exists(self, zone: str, data_type: str, date: date) -> bool:
        """Check if data exists for a given date."""

    @abstractmethod
    def delete(self, zone: str, data_type: str, date: date) -> None:
        """Delete data for a given date."""


class LocalStorage(StorageBackend):
    """
    Filesystem-based storage (development/testing).

    Partition scheme: data/local_lake/{zone}/type={data_type}/year={YYYY}/month={MM}/day={DD}/data.parquet
    """

    def __init__(self, root_path: Optional[Path] = None):
        """
        Initialize LocalStorage.

        Args:
            root_path: Root directory for data lake. Defaults to config.SAMPLE_DATA_PATH
        """
        self.root_path = root_path or settings.DATA_PATH.parent / "data" / "local_lake"
        self.root_path = Path(self.root_path)
        logger.info(f"LocalStorage initialized at {self.root_path}")

    def _get_partition_path(self, zone: str, data_type: str, date: date) -> Path:
        """Construct partitioned path for a data file."""
        return self.root_path / zone / f"type={data_type}" / f"year={date.year}" / f"month={date.month:02d}" / f"day={date.day:02d}"

    def _get_data_path(self, zone: str, data_type: str, date: date) -> Path:
        """Get full path to data.parquet file."""
        return self._get_partition_path(zone, data_type, date) / "data.parquet"

    def write_parquet(
        self,
        df: pd.DataFrame,
        zone: str,
        data_type: str,
        date: date,
    ) -> str:
        """Write DataFrame to local Parquet file with partitioning."""
        partition_path = self._get_partition_path(zone, data_type, date)
        partition_path.mkdir(parents=True, exist_ok=True)

        parquet_path = self._get_data_path(zone, data_type, date)

        # Write with snappy compression
        table = pq.Table.from_pandas(df)
        pq.write_table(table, str(parquet_path), compression="snappy")

        logger.debug(f"Wrote {len(df)} rows to {parquet_path}")
        return str(parquet_path)

    def read_parquet(
        self,
        zone: str,
        data_type: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Read Parquet files within date range, concatenate them."""
        zone_path = self.root_path / zone / f"type={data_type}"

        if not zone_path.exists():
            logger.warning(f"Zone path does not exist: {zone_path}")
            return pd.DataFrame()

        # Find all matching partitions
        parquet_files = sorted(zone_path.glob(f"year=*/month=*/day=*/data.parquet"))

        # Filter by date range
        dfs = []
        for parquet_file in parquet_files:
            # Parse date from path: year=2024/month=01/day=15
            parts = parquet_file.parent.parent.parent.name.split("=")[1]
            year = int(parts)
            month = int(parquet_file.parent.parent.name.split("=")[1])
            day = int(parquet_file.parent.name.split("=")[1])

            file_date = pd.Timestamp(year, month, day).date()

            if start_date <= file_date <= end_date:
                try:
                    df = pq.read_table(str(parquet_file)).to_pandas()
                    dfs.append(df)
                    logger.debug(f"Read {len(df)} rows from {parquet_file}")
                except Exception as e:
                    logger.error(f"Failed to read {parquet_file}: {e}")

        if not dfs:
            logger.warning(f"No data found for {zone}/{data_type} in range {start_date} to {end_date}")
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Read {len(result)} total rows from {len(dfs)} files")
        return result

    def list_available_dates(self, zone: str, data_type: str) -> list[date]:
        """List all available dates for a zone/type."""
        zone_path = self.root_path / zone / f"type={data_type}"

        if not zone_path.exists():
            return []

        dates = []
        for parquet_file in zone_path.glob(f"year=*/month=*/day=*/data.parquet"):
            year = int(parquet_file.parent.parent.parent.name.split("=")[1])
            month = int(parquet_file.parent.parent.name.split("=")[1])
            day = int(parquet_file.parent.name.split("=")[1])
            dates.append(pd.Timestamp(year, month, day).date())

        return sorted(dates)

    def exists(self, zone: str, data_type: str, date: date) -> bool:
        """Check if data exists for given date."""
        return self._get_data_path(zone, data_type, date).exists()

    def delete(self, zone: str, data_type: str, date: date) -> None:
        """Delete data for given date."""
        parquet_path = self._get_data_path(zone, data_type, date)
        if parquet_path.exists():
            parquet_path.unlink()
            logger.info(f"Deleted {parquet_path}")


class S3Storage(StorageBackend):
    """
    AWS S3-based storage (production).

    Partition scheme: s3://{bucket}/{zone}/type={data_type}/year={YYYY}/month={MM}/day={DD}/data.parquet
    """

    def __init__(
        self,
        bucket: str = settings.S3_BUCKET,
        region: str = settings.AWS_REGION,
    ):
        """
        Initialize S3Storage.

        Args:
            bucket: S3 bucket name
            region: AWS region
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for S3Storage. Install with: pip install boto3")

        self.bucket = bucket
        self.region = region
        self.s3_client = boto3.client("s3", region_name=region)
        logger.info(f"S3Storage initialized: s3://{bucket}")

    def _get_s3_key(self, zone: str, data_type: str, date: date) -> str:
        """Construct S3 object key."""
        return f"{zone}/type={data_type}/year={date.year}/month={date.month:02d}/day={date.day:02d}/data.parquet"

    def write_parquet(
        self,
        df: pd.DataFrame,
        zone: str,
        data_type: str,
        date: date,
    ) -> str:
        """Write DataFrame to S3 Parquet."""
        try:
            import io
        except ImportError:
            import io

        s3_key = self._get_s3_key(zone, data_type, date)

        # Write to in-memory buffer
        buffer = io.BytesIO()
        table = pq.Table.from_pandas(df)
        pq.write_table(table, buffer, compression="snappy")
        buffer.seek(0)

        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        s3_path = f"s3://{self.bucket}/{s3_key}"
        logger.debug(f"Wrote {len(df)} rows to {s3_path}")
        return s3_path

    def read_parquet(
        self,
        zone: str,
        data_type: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Read Parquet files from S3 within date range."""
        try:
            import io
        except ImportError:
            import io

        prefix = f"{zone}/type={data_type}/"

        # List objects matching prefix
        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

        if "Contents" not in response:
            logger.warning(f"No objects found with prefix {prefix}")
            return pd.DataFrame()

        dfs = []
        for obj in response["Contents"]:
            key = obj["Key"]

            # Parse date from key: zone/type={type}/year={y}/month={m}/day={d}/data.parquet
            try:
                parts = key.split("/")
                year = int(parts[2].split("=")[1])
                month = int(parts[3].split("=")[1])
                day = int(parts[4].split("=")[1])
                file_date = pd.Timestamp(year, month, day).date()

                if start_date <= file_date <= end_date:
                    # Download and read
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                    df = pq.read_table(response["Body"]).to_pandas()
                    dfs.append(df)
                    logger.debug(f"Read {len(df)} rows from s3://{self.bucket}/{key}")
            except Exception as e:
                logger.error(f"Failed to read s3://{self.bucket}/{key}: {e}")

        if not dfs:
            logger.warning(f"No data found for {zone}/{data_type} in range {start_date} to {end_date}")
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Read {len(result)} total rows from {len(dfs)} S3 objects")
        return result

    def list_available_dates(self, zone: str, data_type: str) -> list[date]:
        """List all available dates for a zone/type."""
        prefix = f"{zone}/type={data_type}/"

        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

        if "Contents" not in response:
            return []

        dates = []
        for obj in response["Contents"]:
            key = obj["Key"]
            try:
                parts = key.split("/")
                year = int(parts[2].split("=")[1])
                month = int(parts[3].split("=")[1])
                day = int(parts[4].split("=")[1])
                dates.append(pd.Timestamp(year, month, day).date())
            except (IndexError, ValueError):
                pass

        return sorted(set(dates))

    def exists(self, zone: str, data_type: str, date: date) -> bool:
        """Check if object exists in S3."""
        s3_key = self._get_s3_key(zone, data_type, date)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception:
            return False

    def delete(self, zone: str, data_type: str, date: date) -> None:
        """Delete object from S3."""
        s3_key = self._get_s3_key(zone, data_type, date)
        self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
        logger.info(f"Deleted s3://{self.bucket}/{s3_key}")


def get_storage(mode: Optional[StorageMode] = None) -> StorageBackend:
    """
    Factory function to get storage backend.

    Args:
        mode: Storage mode (local or s3). Defaults to settings.STORAGE_MODE

    Returns:
        StorageBackend instance (LocalStorage or S3Storage)
    """
    if mode is None:
        mode = settings.STORAGE_MODE

    if mode == StorageMode.LOCAL:
        return LocalStorage()
    elif mode == StorageMode.S3:
        return S3Storage()
    else:
        raise ValueError(f"Unknown storage mode: {mode}")

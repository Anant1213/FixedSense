"""Data storage layer with pluggable backends (local filesystem or S3)."""

from data.storage.s3_client import S3Storage, LocalStorage, StorageBackend, get_storage

__all__ = ["StorageBackend", "LocalStorage", "S3Storage", "get_storage"]

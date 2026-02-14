"""
Streaming module: High-performance data loading from multiple sources.

Public API:
- StreamingDatasetAdapter: Main dataset class
- SmartLRUCache: LRU cache with compression
- DataSource: Abstract base for data sources
- AsyncPrefetcher: Async prefetch worker
"""

from aiprod_trainer.streaming.adapter import StreamingDatasetAdapter, create_streaming_dataset_from_config
from aiprod_trainer.streaming.cache import AsyncPrefetcher, CacheMetrics, SmartLRUCache
from aiprod_trainer.streaming.sources import (
    DataSource,
    DataSourceConfig,
    GCSDataSource,
    HuggingFaceDataSource,
    LocalDataSource,
    S3DataSource,
    create_data_source,
)

__all__ = [
    "StreamingDatasetAdapter",
    "create_streaming_dataset_from_config",
    "SmartLRUCache",
    "AsyncPrefetcher",
    "CacheMetrics",
    "DataSource",
    "DataSourceConfig",
    "LocalDataSource",
    "HuggingFaceDataSource",
    "S3DataSource",
    "GCSDataSource",
    "create_data_source",
]

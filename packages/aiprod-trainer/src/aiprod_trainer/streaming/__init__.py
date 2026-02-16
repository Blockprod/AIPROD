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
    LocalDataSource,
    create_data_source,
)

# Cloud sources — available when aiprod-cloud is installed
try:
    from aiprod_cloud.cloud_sources import (  # noqa: PLC0415
        GCSDataSource,
        HuggingFaceDataSource,
        S3DataSource,
    )
except ImportError:
    GCSDataSource = None  # type: ignore[assignment,misc]
    HuggingFaceDataSource = None  # type: ignore[assignment,misc]
    S3DataSource = None  # type: ignore[assignment,misc]

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

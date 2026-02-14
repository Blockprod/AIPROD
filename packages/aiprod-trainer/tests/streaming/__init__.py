"""
Tests for streaming module - comprehensive test suite.

Organization:
- conftest.py: Shared fixtures and utilities
- test_sources.py: DataSource implementations (local, HF, S3, GCS)
- test_cache.py: SmartLRUCache functionality
- test_prefetcher.py: AsyncPrefetcher behavior
- test_adapter.py: StreamingDatasetAdapter integration
- test_performance.py: Benchmarks and performance tests
"""

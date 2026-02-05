"""
Performance optimization module - Caching, query optimization, async processing
"""

from src.performance.caching_service import get_caching_service, CachingService, CacheBackend
from src.performance.query_optimizer import get_query_optimizer, QueryOptimizer
from src.performance.performance_profiler import get_performance_profiler, PerformanceProfiler
from src.performance.async_processor import get_async_processor, AsyncTaskProcessor
from src.performance.compression_middleware import CompressionMiddleware, CacheHeaderMiddleware
from src.performance.performance_routes import setup_performance_routes

__all__ = [
    "get_caching_service",
    "CachingService",
    "CacheBackend",
    "get_query_optimizer",
    "QueryOptimizer",
    "get_performance_profiler",
    "PerformanceProfiler",
    "get_async_processor",
    "AsyncTaskProcessor",
    "CompressionMiddleware",
    "CacheHeaderMiddleware",
    "setup_performance_routes",
]

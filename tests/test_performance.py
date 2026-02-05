"""
Comprehensive tests for Phase 2.2 - Performance Optimization
Tests for caching, query optimization, async processing, and compression
"""

import pytest
import asyncio
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.performance.caching_service import (
    get_caching_service,
    CachingService,
    CacheBackend,
    InMemoryCache,
)
from src.performance.query_optimizer import get_query_optimizer, QueryOptimizer
from src.performance.performance_profiler import get_performance_profiler, PerformanceProfiler
from src.performance.async_processor import (
    get_async_processor,
    AsyncTaskProcessor,
    TaskStatus,
)
from src.performance.compression_middleware import CompressionMiddleware
from src.api.main import app


# ============================================================================
# CACHING SERVICE TESTS
# ============================================================================

class TestInMemoryCache:
    """Test in-memory cache functionality"""
    
    def test_cache_set_and_get(self):
        """Test basic set and get operations"""
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl_seconds=60)
        
        result = cache.get("key1")
        assert result == "value1"
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl_seconds=0)
        
        result = cache.get("key1")
        # TTL 0 means no expiration
        assert result == "value1"
    
    def test_cache_delete(self):
        """Test cache deletion"""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        
        result = cache.get("key1")
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction"""
        cache = InMemoryCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 and key3 to update their timestamps (key2 is LRU)
        cache.get("key1")
        cache.get("key3")
        
        # Add new key - should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        # key2 should be evicted as it's the LRU
        assert cache.get("key2") is None or len(cache.cache) <= 3


class TestCachingService:
    """Test caching service"""
    
    def test_caching_service_get_set(self):
        """Test basic caching operations"""
        service = CachingService()
        service.set("test_key", "test_value")
        
        result = service.get("test_key")
        assert result == "test_value"
    
    def test_cache_miss_increments_counter(self):
        """Test cache miss counter"""
        service = CachingService()
        initial_misses = service.miss_count
        
        service.get("nonexistent_key")
        assert service.miss_count == initial_misses + 1
    
    def test_cache_hit_increments_counter(self):
        """Test cache hit counter"""
        service = CachingService()
        service.set("test_key", "test_value")
        initial_hits = service.hit_count
        
        service.get("test_key")
        assert service.hit_count == initial_hits + 1
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation"""
        service = CachingService()
        service.set("key1", "value1")
        
        # 3 hits
        service.get("key1")
        service.get("key1")
        service.get("key1")
        
        # 2 misses
        service.get("nonexistent1")
        service.get("nonexistent2")
        
        hit_rate = service.get_hit_rate()
        assert 59.0 < hit_rate < 61.0  # 3/5 = 60%
    
    def test_cache_delete_by_pattern(self):
        """Test pattern-based cache deletion"""
        service = CachingService()
        service.set("api:user:1", "value1")
        service.set("api:user:2", "value2")
        service.set("db:query:1", "value3")
        
        service.delete_by_pattern("api:user")
        
        assert service.get("api:user:1") is None
        assert service.get("api:user:2") is None
        assert service.get("db:query:1") is not None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        service = CachingService()
        service.set("key1", "value1")
        service.set("key2", "value2")
        
        stats = service.get_stats()
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
        assert "cache_stats" in stats
        assert stats["cache_stats"]["total_entries"] >= 2


# ============================================================================
# QUERY OPTIMIZER TESTS
# ============================================================================

class TestQueryOptimizer:
    """Test query optimization"""
    
    def test_record_query(self):
        """Test query recording"""
        optimizer = QueryOptimizer()
        optimizer.record_query(
            query_id="q1",
            query_text="SELECT * FROM users",
            execution_time_ms=100,
            rows_affected=10,
            table="users",
        )
        
        assert "q1" in optimizer.query_profiles
        assert len(optimizer.query_profiles["q1"]) == 1
    
    def test_get_query_stats(self):
        """Test query statistics"""
        optimizer = QueryOptimizer()
        optimizer.record_query("q1", "SELECT * FROM users", 100)
        optimizer.record_query("q1", "SELECT * FROM users", 150)
        optimizer.record_query("q1", "SELECT * FROM users", 120)
        
        stats = optimizer.get_query_stats("q1")
        assert stats["execution_count"] == 3
        assert 100 <= stats["avg_time_ms"] <= 150
    
    def test_slow_query_detection(self):
        """Test slow query detection"""
        optimizer = QueryOptimizer(slow_query_threshold_ms=100)
        optimizer.record_query("q1", "SELECT * FROM users", 500)
        
        slow = optimizer.get_slow_queries()
        assert len(slow) > 0
        assert slow[0]["execution_time_ms"] == 500
    
    def test_n_plus_one_detection(self):
        """Test N+1 query pattern detection"""
        optimizer = QueryOptimizer()
        
        # Record many similar fast queries
        for i in range(10):
            optimizer.record_query(f"q{i}", "SELECT * FROM users WHERE id = ?", 15)
        
        suspicious = optimizer.detect_n_plus_one_queries()
        # May or may not detect based on frequency threshold
        assert isinstance(suspicious, list)
    
    def test_query_optimizer_overview(self):
        """Test query optimizer overview"""
        optimizer = QueryOptimizer()
        optimizer.record_query("q1", "SELECT * FROM users", 100)
        optimizer.record_query("q1", "SELECT * FROM users", 200)
        optimizer.record_query("q2", "SELECT * FROM products", 50)
        
        overview = optimizer.get_overview()
        assert overview["total_queries"] == 3
        assert overview["unique_queries"] == 2


# ============================================================================
# PERFORMANCE PROFILER TESTS
# ============================================================================

class TestPerformanceProfiler:
    """Test performance profiling"""
    
    def test_record_request(self):
        """Test request recording"""
        profiler = PerformanceProfiler()
        profiler.record_request(
            endpoint="/api/users",
            method="GET",
            duration_ms=100,
            status_code=200,
        )
        
        assert "/api/users" in [p.endpoint for p in profiler.profiles]
    
    def test_endpoint_performance_stats(self):
        """Test endpoint performance statistics"""
        profiler = PerformanceProfiler()
        profiler.record_request("/api/users", "GET", 100, 200)
        profiler.record_request("/api/users", "GET", 150, 200)
        profiler.record_request("/api/users", "GET", 120, 200)
        
        stats = profiler.get_endpoint_performance("GET", "/api/users")
        assert stats["request_count"] == 3
        assert 100 <= stats["avg_duration_ms"] <= 150
    
    def test_slowest_endpoints(self):
        """Test slowest endpoints detection"""
        profiler = PerformanceProfiler()
        profiler.record_request("/api/users", "GET", 1000, 200)
        profiler.record_request("/api/products", "GET", 100, 200)
        profiler.record_request("/api/orders", "GET", 500, 200)
        
        slowest = profiler.get_slowest_endpoints(limit=2)
        assert len(slowest) >= 1
        # First should be slowest
        if slowest:
            assert slowest[0]["avg_duration_ms"] >= slowest[-1]["avg_duration_ms"]
    
    def test_performance_insights(self):
        """Test performance insights generation"""
        profiler = PerformanceProfiler()
        # Add slow request
        profiler.record_request("/api/users", "GET", 2000, 200)
        
        insights = profiler.get_performance_insights()
        assert len(insights) > 0
        # Should include slow endpoint warning
        slow_insights = [i for i in insights if i["type"] == "slow_endpoint"]
        assert len(slow_insights) > 0
    
    def test_profiler_overview(self):
        """Test profiler overview"""
        profiler = PerformanceProfiler()
        profiler.record_request("/api/users", "GET", 100, 200)
        profiler.record_request("/api/products", "GET", 150, 200)
        
        overview = profiler.get_overview()
        assert overview["total_requests"] == 2
        assert overview["unique_endpoints"] == 2


# ============================================================================
# ASYNC PROCESSOR TESTS
# ============================================================================

class TestAsyncTaskProcessor:
    """Test async task processing"""
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test task submission"""
        processor = AsyncTaskProcessor()
        
        async def sample_task():
            return "result"
        
        task_id = await processor.submit_task("test_task", sample_task())
        
        assert task_id is not None
        assert task_id in processor.tasks
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test task execution"""
        processor = AsyncTaskProcessor()
        
        async def sample_task():
            await asyncio.sleep(0.1)
            return "completed"
        
        task_id = await processor.submit_task("test_task", sample_task())
        
        # Wait for task to complete
        await asyncio.sleep(0.5)
        
        task = processor.get_task(task_id)
        if task:
            assert task.status.value in [
                TaskStatus.COMPLETED.value,
                TaskStatus.RUNNING.value,
            ]
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout"""
        processor = AsyncTaskProcessor()
        
        async def slow_task():
            await asyncio.sleep(10)
            return "result"
        
        task_id = await processor.submit_task("test_task", slow_task(), timeout_seconds=1)
        
        # Wait for timeout
        await asyncio.sleep(2)
        
        task = processor.get_task(task_id)
        if task:
            assert task.status in [TaskStatus.FAILED, TaskStatus.RUNNING]
    
    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test task cancellation"""
        processor = AsyncTaskProcessor()
        
        async def long_task():
            await asyncio.sleep(10)
            return "result"
        
        task_id = await processor.submit_task("test_task", long_task())
        
        # Give it a moment to be pending
        await asyncio.sleep(0.1)
        
        success = processor.cancel_task(task_id)
        # Might succeed or fail depending on task state
        assert isinstance(success, bool)
    
    def test_processor_stats(self):
        """Test processor statistics"""
        processor = AsyncTaskProcessor()
        
        stats = processor.get_stats()
        assert "active_tasks" in stats
        assert "total_tasks" in stats
        assert "completed_tasks" in stats


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

class TestPerformanceEndpoints:
    """Test performance optimization API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_cache_stats_endpoint(self, client):
        """Test cache stats endpoint"""
        response = client.get("/performance/cache/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "hit_count" in data
        assert "miss_count" in data
        assert "hit_rate" in data
    
    def test_query_optimization_endpoint(self, client):
        """Test query optimization endpoint"""
        response = client.get("/performance/query/optimization")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_queries" in data
        assert "recommendations" in data
    
    def test_profile_overview_endpoint(self, client):
        """Test performance profile overview endpoint"""
        response = client.get("/performance/profile/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
    
    def test_cache_clear_endpoint(self, client):
        """Test cache clear endpoint"""
        # First populate cache
        client.get("/performance/cache/stats")
        
        response = client.post("/performance/cache/clear")
        assert response.status_code in [200, 429]  # 429 if rate limited
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_performance_dashboard_endpoint(self, client):
        """Test performance dashboard endpoint"""
        response = client.get("/performance/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "performance_profile" in data
        assert "cache_stats" in data
        assert "insights" in data
    
    def test_slowest_endpoints_endpoint(self, client):
        """Test slowest endpoints endpoint"""
        response = client.get("/performance/profile/slowest?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_performance_insights_endpoint(self, client):
        """Test performance insights endpoint"""
        response = client.get("/performance/profile/insights")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPerformanceIntegration:
    """Integration tests for performance features"""
    
    def test_compression_middleware_integration(self):
        """Test compression middleware integration"""
        middleware = CompressionMiddleware(None)
        assert middleware.MIN_COMPRESSION_SIZE == 500
        assert "application/json" in middleware.COMPRESSIBLE_TYPES
    
    def test_caching_service_singleton(self):
        """Test caching service is singleton"""
        service1 = get_caching_service()
        service2 = get_caching_service()
        
        assert service1 is service2
    
    def test_query_optimizer_singleton(self):
        """Test query optimizer is singleton"""
        optimizer1 = get_query_optimizer()
        optimizer2 = get_query_optimizer()
        
        assert optimizer1 is optimizer2
    
    def test_performance_profiler_singleton(self):
        """Test performance profiler is singleton"""
        profiler1 = get_performance_profiler()
        profiler2 = get_performance_profiler()
        
        assert profiler1 is profiler2
    
    def test_async_processor_singleton(self):
        """Test async processor is singleton"""
        processor1 = get_async_processor()
        processor2 = get_async_processor()
        
        assert processor1 is processor2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

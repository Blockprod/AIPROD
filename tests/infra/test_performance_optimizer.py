"""
Comprehensive tests for Phase 3.2 - Performance Optimization
"""

import pytest
from datetime import datetime
from src.infra.performance_optimizer import (
    OptimizationType,
    PerformanceMetric,
    OptimizationRecommendation,
    PerformanceProfiler,
    PerformanceBenchmark
)


class TestPerformanceMetric:
    """Test PerformanceMetric"""
    
    def test_metric_creation(self):
        """Test creating a performance metric"""
        metric = PerformanceMetric(
            name="response_time",
            value=450,
            unit="ms"
        )
        assert metric.name == "response_time"
        assert metric.value == 450
        assert metric.unit == "ms"
    
    def test_metric_with_threshold(self):
        """Test metric with threshold"""
        metric = PerformanceMetric(
            name="response_time",
            value=450,
            unit="ms",
            threshold=500
        )
        assert metric.threshold == 500
    
    def test_metric_timestamp(self):
        """Test metric has timestamp"""
        metric = PerformanceMetric(
            name="test",
            value=100,
            unit="ms"
        )
        assert metric.timestamp is not None
        assert isinstance(metric.timestamp, datetime)


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation"""
    
    def test_recommendation_creation(self):
        """Test creating a recommendation"""
        rec = OptimizationRecommendation(
            area="database",
            title="Optimize queries",
            description="Add indexes",
            priority="HIGH",
            effort="MEDIUM",
            expected_impact="20% improvement",
            estimated_hours=4
        )
        assert rec.area == "database"
        assert rec.priority == "HIGH"
        assert rec.estimated_hours == 4
    
    def test_recommendation_attributes(self):
        """Test all recommendation attributes"""
        rec = OptimizationRecommendation(
            area="caching",
            title="Improve cache",
            description="Increase TTL",
            priority="MEDIUM",
            effort="LOW",
            expected_impact="15% improvement",
            estimated_hours=2
        )
        assert rec.area == "caching"
        assert rec.title == "Improve cache"
        assert rec.effort == "LOW"


class TestPerformanceProfiler:
    """Test PerformanceProfiler"""
    
    def test_profiler_initialization(self):
        """Test profiler initialization"""
        profiler = PerformanceProfiler()
        assert profiler.metrics == []
        assert profiler.profile_timestamp is not None
    
    def test_record_single_metric(self):
        """Test recording a single metric"""
        profiler = PerformanceProfiler()
        metric = profiler.record_metric("response_time", 450, "ms", threshold=500)
        assert len(profiler.metrics) == 1
        assert metric.name == "response_time"
    
    def test_record_multiple_metrics(self):
        """Test recording multiple metrics"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 450, "ms", threshold=500)
        profiler.record_metric("response_time", 480, "ms", threshold=500)
        profiler.record_metric("memory_usage", 1800, "mb", threshold=2048)
        assert len(profiler.metrics) == 3
    
    def test_analyze_empty_metrics(self):
        """Test analyzing with no metrics"""
        profiler = PerformanceProfiler()
        analysis = profiler.analyze()
        assert "status" in analysis
        assert analysis["status"] == "No metrics recorded"
    
    def test_analyze_metrics_basic(self):
        """Test basic metric analysis"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 450, "ms", threshold=500)
        profiler.record_metric("response_time", 480, "ms", threshold=500)
        
        analysis = profiler.analyze()
        assert analysis["total_metrics"] == 2
        assert "response_time" in analysis["averages"]
        assert 460 <= analysis["averages"]["response_time"] <= 470
    
    def test_analyze_exceeding_threshold(self):
        """Test detecting metrics exceeding thresholds"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 450, "ms", threshold=400)
        profiler.record_metric("response_time", 480, "ms", threshold=400)
        
        analysis = profiler.analyze()
        assert len(analysis["exceeding_threshold"]) > 0
        exceeded = analysis["exceeding_threshold"][0]
        assert exceeded["metric"] == "response_time"
        assert exceeded["threshold"] == 400
    
    def test_analyze_within_threshold(self):
        """Test metrics within threshold"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 450, "ms", threshold=500)
        profiler.record_metric("response_time", 480, "ms", threshold=500)
        
        analysis = profiler.analyze()
        assert len(analysis["exceeding_threshold"]) == 0
    
    def test_analyze_statistics(self):
        """Test statistics in analysis"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 400, "ms")
        profiler.record_metric("response_time", 500, "ms")
        profiler.record_metric("response_time", 600, "ms")
        
        analysis = profiler.analyze()
        stats = analysis["statistics"]["response_time"]
        assert stats["min"] == 400
        assert stats["max"] == 600
        assert stats["count"] == 3
    
    def test_analyze_overage_percent(self):
        """Test overage percentage calculation"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 500, "ms", threshold=400)
        
        analysis = profiler.analyze()
        exceeded = analysis["exceeding_threshold"][0]
        assert exceeded["overage_percent"] == 25.0  # (500-400)/400 * 100
    
    def test_generate_recommendations_response_time(self):
        """Test recommendations for response time"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 550, "ms", threshold=500)
        
        recommendations = profiler.generate_recommendations()
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert "optimize" in rec.title.lower()
    
    def test_generate_recommendations_memory(self):
        """Test recommendations for memory issues"""
        profiler = PerformanceProfiler()
        profiler.record_metric("memory_usage", 2500, "mb", threshold=2048)
        
        recommendations = profiler.generate_recommendations()
        assert len(recommendations) > 0
        assert any("memory" in r.description.lower() for r in recommendations)
    
    def test_generate_recommendations_cache(self):
        """Test recommendations for cache issues"""
        profiler = PerformanceProfiler()
        profiler.record_metric("cache_performance", 150, "ms", threshold=100)
        
        recommendations = profiler.generate_recommendations()
        assert len(recommendations) > 0
    
    def test_generate_no_recommendations(self):
        """Test no recommendations when all metrics OK"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 400, "ms", threshold=500)
        profiler.record_metric("memory_usage", 1800, "mb", threshold=2048)
        
        recommendations = profiler.generate_recommendations()
        assert len(recommendations) == 0
    
    def test_recommendation_priority(self):
        """Test recommendation has correct priority"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 600, "ms", threshold=500)
        
        recommendations = profiler.generate_recommendations()
        assert recommendations[0].priority == "HIGH"
    
    def test_recommendation_effort_estimate(self):
        """Test recommendation has effort estimate"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 600, "ms", threshold=500)
        
        recommendations = profiler.generate_recommendations()
        assert recommendations[0].estimated_hours > 0
    
    def test_multiple_metric_groups(self):
        """Test analyzing multiple metric types"""
        profiler = PerformanceProfiler()
        profiler.record_metric("response_time", 400, "ms")
        profiler.record_metric("memory_usage", 1800, "mb")
        profiler.record_metric("cache_hits", 95, "percent")
        
        analysis = profiler.analyze()
        assert len(analysis["averages"]) == 3


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark"""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        bench = PerformanceBenchmark()
        assert len(bench.benchmarks) == 5
        assert "api_response_time_p99_ms" in bench.benchmarks
    
    def test_benchmark_update(self):
        """Test updating benchmark value"""
        bench = PerformanceBenchmark()
        result = bench.update_benchmark("api_response_time_p99_ms", 450)
        assert result is True
        assert bench.benchmarks["api_response_time_p99_ms"]["current"] == 450
    
    def test_benchmark_update_invalid(self):
        """Test updating non-existent benchmark"""
        bench = PerformanceBenchmark()
        result = bench.update_benchmark("invalid_benchmark", 100)
        assert result is False
    
    def test_benchmark_status_pending(self):
        """Test benchmark status when pending"""
        bench = PerformanceBenchmark()
        status = bench.get_benchmark_status("api_response_time_p99_ms")
        assert status is not None
        if status is not None:
            assert status["status"] == "PENDING"
    
    def test_benchmark_status_pass_response_time(self):
        """Test benchmark passes for good response time"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("api_response_time_p99_ms", 450)
        status = bench.get_benchmark_status("api_response_time_p99_ms")
        assert status is not None
        if status is not None:
            assert status["meets_target"] is True
            assert status["status"] == "PASS"
    
    def test_benchmark_status_fail_response_time(self):
        """Test benchmark fails for slow response time"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("api_response_time_p99_ms", 600)
        status = bench.get_benchmark_status("api_response_time_p99_ms")
        assert status is not None
        if status is not None:
            assert status["meets_target"] is False
            assert status["status"] == "FAIL"
    
    def test_benchmark_status_pass_throughput(self):
        """Test benchmark passes for good throughput"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("throughput_rps", 1100)
        status = bench.get_benchmark_status("throughput_rps")
        assert status is not None
        if status is not None:
            assert status["meets_target"] is True
            assert status["status"] == "PASS"
    
    def test_benchmark_status_fail_throughput(self):
        """Test benchmark fails for low throughput"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("throughput_rps", 800)
        status = bench.get_benchmark_status("throughput_rps")
        assert status is not None
        if status is not None:
            assert status["meets_target"] is False
            assert status["status"] == "FAIL"
    
    def test_benchmark_status_pass_cache_hit(self):
        """Test benchmark passes for good cache hit rate"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("cache_hit_rate_percent", 90)
        status = bench.get_benchmark_status("cache_hit_rate_percent")
        assert status is not None
        assert status["meets_target"] is True
    
    def test_benchmark_invalid_name(self):
        """Test getting status for invalid benchmark"""
        bench = PerformanceBenchmark()
        status = bench.get_benchmark_status("invalid")
        assert status is None
    
    def test_get_all_benchmarks(self):
        """Test getting all benchmarks"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("api_response_time_p99_ms", 450)
        bench.update_benchmark("cache_hit_rate_percent", 90)
        
        all_benchmarks = bench.get_all_benchmarks()
        assert "timestamp" in all_benchmarks
        assert "benchmarks" in all_benchmarks
        assert len(all_benchmarks["benchmarks"]) == 5
    
    def test_all_benchmarks_overall_pass(self):
        """Test overall pass when all benchmarks pass"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("api_response_time_p99_ms", 450)
        bench.update_benchmark("database_query_time_p99_ms", 90)
        bench.update_benchmark("cache_hit_rate_percent", 90)
        bench.update_benchmark("memory_usage_mb", 1800)
        bench.update_benchmark("throughput_rps", 1100)
        
        all_benchmarks = bench.get_all_benchmarks()
        assert all_benchmarks["overall_status"] == "PASS"
    
    def test_all_benchmarks_overall_fail(self):
        """Test overall fail when any benchmark fails"""
        bench = PerformanceBenchmark()
        bench.update_benchmark("api_response_time_p99_ms", 600)  # Fail
        bench.update_benchmark("database_query_time_p99_ms", 90)
        bench.update_benchmark("cache_hit_rate_percent", 90)
        bench.update_benchmark("memory_usage_mb", 1800)
        bench.update_benchmark("throughput_rps", 1100)
        
        all_benchmarks = bench.get_all_benchmarks()
        assert all_benchmarks["overall_status"] == "FAIL"
    
    def test_benchmark_with_correct_targets(self):
        """Test benchmarks have correct target values"""
        bench = PerformanceBenchmark()
        assert bench.benchmarks["api_response_time_p99_ms"]["target"] == 500
        assert bench.benchmarks["database_query_time_p99_ms"]["target"] == 100
        assert bench.benchmarks["cache_hit_rate_percent"]["target"] == 85
        assert bench.benchmarks["memory_usage_mb"]["target"] == 2048
        assert bench.benchmarks["throughput_rps"]["target"] == 1000


class TestOptimizationType:
    """Test OptimizationType enum"""
    
    def test_optimization_types_exist(self):
        """Test all optimization types are defined"""
        assert OptimizationType.DATABASE.value == "database"
        assert OptimizationType.CACHING.value == "caching"
        assert OptimizationType.QUERY.value == "query"
        assert OptimizationType.MEMORY.value == "memory"
        assert OptimizationType.NETWORK.value == "network"


class TestPerformanceIntegration:
    """Integration tests for performance optimization"""
    
    def test_profiler_to_benchmark_workflow(self):
        """Test workflow from profiling to benchmarking"""
        # Profile
        profiler = PerformanceProfiler()
        profiler.record_metric("api_response_time_p99_ms", 450, "ms", threshold=500)
        profiler.record_metric("api_response_time_p99_ms", 480, "ms", threshold=500)
        
        # Analyze
        analysis = profiler.analyze()
        avg_response_time = analysis["averages"]["api_response_time_p99_ms"]
        
        # Benchmark
        bench = PerformanceBenchmark()
        bench.update_benchmark("api_response_time_p99_ms", avg_response_time)
        status = bench.get_benchmark_status("api_response_time_p99_ms")
        
        assert status is not None
        assert status["meets_target"] is True
    
    def test_performance_degradation_detection(self):
        """Test detecting performance degradation"""
        # Baseline
        profiler1 = PerformanceProfiler()
        profiler1.record_metric("response_time", 400, "ms", threshold=500)
        
        # Degraded
        profiler2 = PerformanceProfiler()
        profiler2.record_metric("response_time", 520, "ms", threshold=500)
        
        analysis1 = profiler1.analyze()
        analysis2 = profiler2.analyze()
        
        assert len(analysis1["exceeding_threshold"]) == 0
        assert len(analysis2["exceeding_threshold"]) == 1
    
    def test_recommendation_generation_flow(self):
        """Test full recommendation generation flow"""
        profiler = PerformanceProfiler()
        
        # Record poor metrics
        profiler.record_metric("response_time", 600, "ms", threshold=500)
        profiler.record_metric("memory_usage", 2500, "mb", threshold=2048)
        
        # Generate recommendations
        recommendations = profiler.generate_recommendations()
        
        assert len(recommendations) >= 2
        assert all(isinstance(r, OptimizationRecommendation) for r in recommendations)
        assert all(r.priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] for r in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

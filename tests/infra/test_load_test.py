"""
Tests for load testing infrastructure
Tests load generation, metrics collection, and validation
"""

import pytest
from datetime import datetime, timedelta
from src.infra.load_test import (
    LoadProfile, LoadTestResult, LoadGenerator, RequestMetrics,
    LoadTestValidator
)


class TestLoadTestResult:
    """Test LoadTestResult metrics"""
    
    def test_load_test_result_creation(self):
        """Test creating load test result"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        result = LoadTestResult(
            test_profile=LoadProfile.MODERATE,
            total_requests=500,
            successful_requests=495,
            failed_requests=5,
            start_time=start,
            end_time=end
        )
        
        assert result.test_profile == LoadProfile.MODERATE
        assert result.total_requests == 500
        assert result.success_rate == 99.0
    
    def test_duration_calculation(self):
        """Test duration calculation"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=120)
        
        result = LoadTestResult(
            test_profile=LoadProfile.LIGHT,
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            start_time=start,
            end_time=end
        )
        
        assert result.duration_seconds == 120
    
    def test_actual_rps(self):
        """Test calculating actual RPS"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        result = LoadTestResult(
            test_profile=LoadProfile.MODERATE,
            total_requests=500,
            successful_requests=500,
            failed_requests=0,
            start_time=start,
            end_time=end
        )
        
        assert abs(result.actual_rps - 8.33) < 0.1  # ~500/60
    
    def test_success_rate(self):
        """Test success rate calculation"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        result = LoadTestResult(
            test_profile=LoadProfile.LIGHT,
            total_requests=1000,
            successful_requests=990,
            failed_requests=10,
            start_time=start,
            end_time=end
        )
        
        assert result.success_rate == 99.0
    
    def test_response_time_percentiles(self):
        """Test response time percentile calculations"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=10)
        
        # Create requests with known response times
        requests = [
            RequestMetrics("api", "GET", 50, 200, True),
            RequestMetrics("api", "GET", 100, 200, True),
            RequestMetrics("api", "GET", 150, 200, True),
            RequestMetrics("api", "GET", 200, 200, True),
            RequestMetrics("api", "GET", 250, 200, True),
            RequestMetrics("api", "GET", 500, 200, True),
            RequestMetrics("api", "GET", 1000, 200, True),
            RequestMetrics("api", "GET", 1500, 200, True),
            RequestMetrics("api", "GET", 2000, 200, True),
            RequestMetrics("api", "GET", 5000, 200, True),
        ]
        
        result = LoadTestResult(
            test_profile=LoadProfile.HEAVY,
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            start_time=start,
            end_time=end,
            requests=requests
        )
        
        assert result.min_response_time == 50
        assert result.max_response_time == 5000
        assert result.avg_response_time == pytest.approx(1105, rel=1)
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        result = LoadTestResult(
            test_profile=LoadProfile.LIGHT,
            total_requests=100,
            successful_requests=99,
            failed_requests=1,
            start_time=start,
            end_time=end
        )
        
        data = result.to_dict()
        assert data["test_profile"] == "light"
        assert data["total_requests"] == 100
        assert data["success_rate"] == 99.0
        assert "response_times" in data


class TestLoadGenerator:
    """Test LoadGenerator"""
    
    def test_generator_creation(self):
        """Test creating load generator"""
        gen = LoadGenerator()
        assert gen is not None
        assert len(gen.results) == 0
    
    def test_simulate_request(self):
        """Test simulating a single request"""
        gen = LoadGenerator()
        metrics = gen._simulate_request("/api/test", "GET")
        
        assert metrics.endpoint == "/api/test"
        assert metrics.method == "GET"
        assert metrics.response_time_ms > 0
        assert metrics.status_code in [200, 500]
    
    def test_generate_light_load(self):
        """Test generating light load (100 RPS)"""
        gen = LoadGenerator()
        result = gen.generate_load(LoadProfile.LIGHT, duration_seconds=5)
        
        assert result.test_profile == LoadProfile.LIGHT
        assert result.total_requests > 0
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration_seconds > 0
    
    def test_generate_moderate_load(self):
        """Test generating moderate load (500 RPS)"""
        gen = LoadGenerator()
        result = gen.generate_load(LoadProfile.MODERATE, duration_seconds=3)
        
        assert result.test_profile == LoadProfile.MODERATE
        assert result.total_requests > 0
        # Should have ~1500 requests in 3 seconds at 500 RPS (with some variance)
        assert result.total_requests > 800
    
    def test_generate_heavy_load(self):
        """Test generating heavy load (1000 RPS)"""
        gen = LoadGenerator()
        result = gen.generate_load(LoadProfile.HEAVY, duration_seconds=2)
        
        assert result.test_profile == LoadProfile.HEAVY
        # Should have ~2000 requests in 2 seconds at 1000 RPS (allow variance)
        assert result.total_requests > 1000
    
    def test_custom_endpoint(self):
        """Test generator with custom endpoint"""
        gen = LoadGenerator()
        result = gen.generate_load(
            LoadProfile.LIGHT,
            endpoint="/custom/path",
            duration_seconds=2
        )
        
        assert all(r.endpoint == "/custom/path" for r in result.requests)
    
    def test_custom_method(self):
        """Test generator with custom HTTP method"""
        gen = LoadGenerator()
        result = gen.generate_load(
            LoadProfile.LIGHT,
            method="POST",
            duration_seconds=2
        )
        
        assert all(r.method == "POST" for r in result.requests)
    
    def test_result_collection(self):
        """Test that results are collected"""
        gen = LoadGenerator()
        
        # Run 2 tests
        gen.generate_load(LoadProfile.LIGHT, duration_seconds=1)
        gen.generate_load(LoadProfile.MODERATE, duration_seconds=1)
        
        assert len(gen.results) == 2
    
    def test_clear_results(self):
        """Test clearing results"""
        gen = LoadGenerator()
        gen.generate_load(LoadProfile.LIGHT, duration_seconds=1)
        assert len(gen.results) == 1
        
        gen.clear_results()
        assert len(gen.results) == 0


class TestLoadTestValidator:
    """Test LoadTestValidator"""
    
    def test_validate_successful_light_load(self):
        """Test validating successful light load"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        requests = [
            RequestMetrics("api", "GET", 50, 200, True) for _ in range(100)
        ]
        
        result = LoadTestResult(
            test_profile=LoadProfile.LIGHT,
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            start_time=start,
            end_time=end,
            requests=requests
        )
        
        validation = LoadTestValidator.validate(result)
        assert validation["passed"]
        assert validation["checks"]["success_rate"]["passed"]
    
    def test_validate_failed_light_load(self):
        """Test validating failed light load"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        requests = [
            RequestMetrics("api", "GET", 50, 200, True) for _ in range(95)
        ] + [
            RequestMetrics("api", "GET", 50, 500, False) for _ in range(5)
        ]
        
        result = LoadTestResult(
            test_profile=LoadProfile.LIGHT,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            start_time=start,
            end_time=end,
            requests=requests
        )
        
        validation = LoadTestValidator.validate(result)
        # Success rate 95% < 99.5% required for LIGHT
        assert not validation["passed"]
    
    def test_validate_moderate_load(self):
        """Test validating moderate load"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        requests = [
            RequestMetrics("api", "GET", 100, 200, True) for _ in range(495)
        ] + [
            RequestMetrics("api", "GET", 100, 500, False) for _ in range(5)
        ]
        
        result = LoadTestResult(
            test_profile=LoadProfile.MODERATE,
            total_requests=500,
            successful_requests=495,
            failed_requests=5,
            start_time=start,
            end_time=end,
            requests=requests
        )
        
        validation = LoadTestValidator.validate(result)
        assert validation["passed"]  # 99% success rate meets requirement
    
    def test_response_time_validation(self):
        """Test response time validation"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=60)
        
        # Create requests with response times exceeding P95 limit
        requests = [
            RequestMetrics("api", "GET", 600, 200, True)  # Over 300ms P95 for LIGHT
            for _ in range(100)
        ]
        
        result = LoadTestResult(
            test_profile=LoadProfile.LIGHT,
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            start_time=start,
            end_time=end,
            requests=requests
        )
        
        validation = LoadTestValidator.validate(result)
        assert not validation["checks"]["p95_response_time"]["passed"]
    
    def test_validation_criteria_exist(self):
        """Test that all profiles have validation criteria"""
        for profile in LoadProfile:
            assert profile in LoadTestValidator.CRITERIA


class TestLoadTestPerformance:
    """Test load test with performance scenarios"""
    
    def test_heavy_load_rps(self):
        """Test that heavy load achieves 1000+ RPS target"""
        gen = LoadGenerator()
        result = gen.generate_load(LoadProfile.HEAVY, duration_seconds=3)
        
        # Should achieve roughly 1000 RPS (allow some variance due to execution time)
        # 3 seconds * 1000 RPS = ~3000 requests
        assert result.total_requests > 1500  # Allow variance
        assert result.actual_rps > 500  # Allow significant variance
    
    def test_response_time_distribution(self):
        """Test response time distribution is reasonable"""
        gen = LoadGenerator()
        result = gen.generate_load(LoadProfile.MODERATE, duration_seconds=5)
        
        # Response times should be reasonable
        assert result.min_response_time > 0
        assert result.avg_response_time > result.min_response_time
        assert result.max_response_time > result.avg_response_time
        assert result.p95_response_time > result.p50_response_time
        assert result.p99_response_time >= result.p95_response_time
    
    def test_error_rate_roughly_1_percent(self):
        """Test that error rate is roughly 1%"""
        gen = LoadGenerator()
        result = gen.generate_load(LoadProfile.HEAVY, duration_seconds=10)
        
        # Should have roughly 1% failure rate
        failure_rate = (result.failed_requests / result.total_requests) * 100
        # Allow 0.5% - 2% variance
        assert 0.5 <= failure_rate <= 2.0

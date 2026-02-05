"""
Load testing infrastructure for AIPROD V33
Tests system performance under high load (1000+ RPS)
Uses simulated load for testing without external dependencies
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import statistics


class LoadProfile(Enum):
    """Load test profiles"""
    LIGHT = "light"          # 100 RPS
    MODERATE = "moderate"    # 500 RPS
    HEAVY = "heavy"          # 1000 RPS
    EXTREME = "extreme"      # 5000 RPS


@dataclass
class RequestMetrics:
    """Metrics for individual request"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


@dataclass
class LoadTestResult:
    """Results of load test"""
    test_profile: LoadProfile
    total_requests: int
    successful_requests: int
    failed_requests: int
    start_time: datetime
    end_time: Optional[datetime] = None
    requests: List[RequestMetrics] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Test duration in seconds"""
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def actual_rps(self) -> float:
        """Actual requests per second"""
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0
    
    @property
    def success_rate(self) -> float:
        """Success rate percentage"""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def p50_response_time(self) -> float:
        """50th percentile response time (ms)"""
        if not self.requests:
            return 0
        times = [r.response_time_ms for r in self.requests]
        return statistics.median(times)
    
    @property
    def p95_response_time(self) -> float:
        """95th percentile response time (ms)"""
        if not self.requests:
            return 0
        times = sorted([r.response_time_ms for r in self.requests])
        idx = int(len(times) * 0.95)
        return times[idx] if idx < len(times) else 0
    
    @property
    def p99_response_time(self) -> float:
        """99th percentile response time (ms)"""
        if not self.requests:
            return 0
        times = sorted([r.response_time_ms for r in self.requests])
        idx = int(len(times) * 0.99)
        return times[idx] if idx < len(times) else 0
    
    @property
    def max_response_time(self) -> float:
        """Maximum response time (ms)"""
        if not self.requests:
            return 0
        return max((r.response_time_ms for r in self.requests), default=0)
    
    @property
    def min_response_time(self) -> float:
        """Minimum response time (ms)"""
        if not self.requests:
            return 0
        return min((r.response_time_ms for r in self.requests), default=0)
    
    @property
    def avg_response_time(self) -> float:
        """Average response time (ms)"""
        if not self.requests:
            return 0
        return statistics.mean([r.response_time_ms for r in self.requests])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_profile": self.test_profile.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "actual_rps": round(self.actual_rps, 2),
            "response_times": {
                "min_ms": round(self.min_response_time, 2),
                "max_ms": round(self.max_response_time, 2),
                "avg_ms": round(self.avg_response_time, 2),
                "p50_ms": round(self.p50_response_time, 2),
                "p95_ms": round(self.p95_response_time, 2),
                "p99_ms": round(self.p99_response_time, 2),
            }
        }


class LoadGenerator:
    """Generates simulated load for testing"""
    
    def __init__(self):
        """Initialize load generator"""
        self.results: List[LoadTestResult] = []
    
    def generate_load(
        self,
        profile: LoadProfile,
        endpoint: str = "/api/jobs",
        method: str = "GET",
        duration_seconds: int = 60,
        request_handler: Optional[Callable] = None
    ) -> LoadTestResult:
        """Generate load against endpoint"""
        
        # Determine target RPS based on profile
        profile_rps = {
            LoadProfile.LIGHT: 100,
            LoadProfile.MODERATE: 500,
            LoadProfile.HEAVY: 1000,
            LoadProfile.EXTREME: 5000,
        }
        target_rps = profile_rps[profile]
        
        # Calculate request interval
        request_interval = 1.0 / target_rps
        
        result = LoadTestResult(
            test_profile=profile,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            start_time=datetime.utcnow(),
            end_time=None
        )
        
        request_count = 0
        start_time = time.time()
        last_request_time = start_time
        
        # Generate requests for specified duration
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Check if it's time for next request
            if current_time - last_request_time >= request_interval:
                # Make request (simulated if no handler provided)
                if request_handler:
                    metrics = request_handler(endpoint, method)
                else:
                    metrics = self._simulate_request(endpoint, method)
                
                result.requests.append(metrics)
                result.total_requests += 1
                request_count += 1
                
                if metrics.success:
                    result.successful_requests += 1
                else:
                    result.failed_requests += 1
                
                last_request_time = current_time
            
            # Small sleep to prevent busy waiting
            time.sleep(0.001)
        
        result.end_time = datetime.utcnow()
        self.results.append(result)
        
        return result
    
    def _simulate_request(self, endpoint: str, method: str) -> RequestMetrics:
        """Simulate a request with realistic response times"""
        import random
        
        # Simulate response time (mostly 50-200ms, some slower)
        if random.random() < 0.95:
            response_time = random.gauss(100, 30)  # Normal distribution
        else:
            response_time = random.gauss(300, 100)  # Slower outliers
        
        response_time = max(10, response_time)  # Min 10ms
        
        # Simulate occasional failures (1% error rate)
        success = random.random() > 0.01
        status_code = 200 if success else 500
        
        return RequestMetrics(
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time,
            status_code=status_code,
            success=success,
            error_message=None if success else "Simulated error"
        )
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all tests"""
        if not self.results:
            return {}
        
        return {
            "total_tests": len(self.results),
            "tests": [r.to_dict() for r in self.results],
            "overall_success_rate": sum(r.success_rate for r in self.results) / len(self.results),
            "total_requests": sum(r.total_requests for r in self.results),
        }
    
    def clear_results(self):
        """Clear test results"""
        self.results.clear()


class LoadTestValidator:
    """Validates load test results against criteria"""
    
    # Acceptance criteria
    CRITERIA = {
        LoadProfile.LIGHT: {
            "success_rate": 99.5,
            "p95_response_time": 300,
            "p99_response_time": 500,
        },
        LoadProfile.MODERATE: {
            "success_rate": 99.0,
            "p95_response_time": 400,
            "p99_response_time": 700,
        },
        LoadProfile.HEAVY: {
            "success_rate": 98.0,
            "p95_response_time": 500,
            "p99_response_time": 1000,
        },
        LoadProfile.EXTREME: {
            "success_rate": 95.0,
            "p95_response_time": 1000,
            "p99_response_time": 2000,
        },
    }
    
    @classmethod
    def validate(cls, result: LoadTestResult) -> Dict[str, Any]:
        """Validate test result against criteria"""
        criteria = cls.CRITERIA.get(result.test_profile, {})
        
        validation = {
            "profile": result.test_profile.value,
            "passed": True,
            "checks": {}
        }
        
        # Check success rate
        if "success_rate" in criteria:
            passed = result.success_rate >= criteria["success_rate"]
            validation["checks"]["success_rate"] = {
                "required": criteria["success_rate"],
                "actual": result.success_rate,
                "passed": passed
            }
            validation["passed"] = validation["passed"] and passed
        
        # Check P95 response time
        if "p95_response_time" in criteria:
            passed = result.p95_response_time <= criteria["p95_response_time"]
            validation["checks"]["p95_response_time"] = {
                "required_max_ms": criteria["p95_response_time"],
                "actual_ms": result.p95_response_time,
                "passed": passed
            }
            validation["passed"] = validation["passed"] and passed
        
        # Check P99 response time
        if "p99_response_time" in criteria:
            passed = result.p99_response_time <= criteria["p99_response_time"]
            validation["checks"]["p99_response_time"] = {
                "required_max_ms": criteria["p99_response_time"],
                "actual_ms": result.p99_response_time,
                "passed": passed
            }
            validation["passed"] = validation["passed"] and passed
        
        return validation

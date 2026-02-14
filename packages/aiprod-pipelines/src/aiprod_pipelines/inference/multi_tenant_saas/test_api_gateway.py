"""
Comprehensive tests for API gateway and rate limiting.

Tests:
  - Rate limiting
  - Request validation
  - API gateway routing
  - Response handling
  - Gateway metrics
  - Usage tracking and metering
"""

import unittest
import time
from datetime import datetime, timedelta

from aiprod_pipelines.inference.multi_tenant_saas.api_gateway import (
    RateLimitConfig,
    RateLimitWindow,
    RateLimiter,
    RequestValidator,
    APIRequest,
    APIResponse,
    APIGateway,
    GatewayMetrics,
)

from aiprod_pipelines.inference.multi_tenant_saas.usage_tracking import (
    UsageEvent,
    UsageEventLogger,
    MeteringEngine,
    UsageAggregator,
    UsageEventType,
)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.limiter = RateLimiter()
    
    def test_within_rate_limit(self):
        """Test request within limit."""
        config = RateLimitConfig(
            requests_per_window=5,
            window_type=RateLimitWindow.MINUTE,
        )
        self.limiter.set_tenant_limit("t1", config)
        
        # First 5 requests should succeed
        for i in range(5):
            allowed, _ = self.limiter.check_rate_limit("t1", "user_1")
            self.assertTrue(allowed)
    
    def test_exceed_rate_limit(self):
        """Test request exceeding limit."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_type=RateLimitWindow.MINUTE,
        )
        self.limiter.set_tenant_limit("t1", config)
        
        # First 2 allowed
        for i in range(2):
            allowed, _ = self.limiter.check_rate_limit("t1", "user_1")
            self.assertTrue(allowed)
        
        # 3rd should be denied
        allowed, info = self.limiter.check_rate_limit("t1", "user_1")
        self.assertFalse(allowed)
    
    def test_burst_tokens(self):
        """Test burst token usage."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_type=RateLimitWindow.MINUTE,
            burst_size=2,
        )
        self.limiter.set_tenant_limit("t1", config)
        
        # Use 2 regular + 2 burst tokens
        for i in range(4):
            allowed, info = self.limiter.check_rate_limit("t1", "user_1")
            self.assertTrue(allowed)
        
        # 5th request should exceed
        allowed, _ = self.limiter.check_rate_limit("t1", "user_1")
        self.assertFalse(allowed)
    
    def test_different_users(self):
        """Test rate limiting per user."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_type=RateLimitWindow.MINUTE,
        )
        self.limiter.set_tenant_limit("t1", config)
        
        # User 1 uses limit
        for i in range(2):
            allowed, _ = self.limiter.check_rate_limit("t1", "user_1")
            self.assertTrue(allowed)
        
        # User 2 should have fresh limit
        allowed, _ = self.limiter.check_rate_limit("t1", "user_2")
        self.assertTrue(allowed)


class TestRequestValidator(unittest.TestCase):
    """Test request validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = RequestValidator()
    
    def test_basic_validation(self):
        """Test basic request validation."""
        request = APIRequest(
            request_id="req_1",
            tenant_id="t1",
            user_id="user_1",
            endpoint="generate_video",
        )
        
        is_valid, errors = self.validator.validate_request(request)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_missing_fields(self):
        """Test validation with missing fields."""
        request = APIRequest(
            request_id="req_1",
            tenant_id="",
            user_id="user_1",
        )
        
        is_valid, errors = self.validator.validate_request(request)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_custom_validator(self):
        """Test custom validator registration."""
        def check_body(req):
            if not req.body or "prompt" not in req.body:
                return False, "prompt is required"
            return True, ""
        
        self.validator.register_validator("generate_video", check_body)
        
        request = APIRequest(
            request_id="req_1",
            tenant_id="t1",
            user_id="user_1",
            endpoint="generate_video",
            body={},
        )
        
        is_valid, errors = self.validator.validate_request(request)
        self.assertFalse(is_valid)
        self.assertIn("prompt is required", errors)


class TestAPIGateway(unittest.TestCase):
    """Test API gateway."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gateway = APIGateway()
    
    def test_register_handler(self):
        """Test handler registration."""
        def handler(req):
            return {"status_code": 200, "message": "Success", "data": {"result": "ok"}}
        
        self.gateway.register_handler("test_endpoint", handler)
        
        request = APIRequest(
            request_id="req_1",
            tenant_id="t1",
            user_id="user_1",
            endpoint="test_endpoint",
        )
        
        response = self.gateway.process_request(request)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.success)
    
    def test_rate_limit_in_gateway(self):
        """Test rate limiting integrated in gateway."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_type=RateLimitWindow.MINUTE,
        )
        self.gateway.rate_limiter.set_tenant_limit("t1", config)
        
        # Register handler
        def handler(req):
            return {"status_code": 200, "message": "OK"}
        
        self.gateway.register_handler("test", handler)
        
        # First 2 requests should succeed
        for i in range(2):
            request = APIRequest(
                request_id=f"req_{i}",
                tenant_id="t1",
                user_id="user_1",
                endpoint="test",
            )
            response = self.gateway.process_request(request)
            self.assertEqual(response.status_code, 200)
        
        # 3rd should be rate limited
        request = APIRequest(
            request_id="req_3",
            tenant_id="t1",
            user_id="user_1",
            endpoint="test",
        )
        response = self.gateway.process_request(request)
        self.assertEqual(response.status_code, 429)  # Too Many Requests
    
    def test_missing_handler(self):
        """Test missing handler."""
        request = APIRequest(
            request_id="req_1",
            tenant_id="t1",
            user_id="user_1",
            endpoint="nonexistent",
        )
        
        response = self.gateway.process_request(request)
        self.assertEqual(response.status_code, 404)
    
    def test_gateway_metrics(self):
        """Test gateway metrics."""
        def handler(req):
            return {"status_code": 200, "message": "OK"}
        
        self.gateway.register_handler("test", handler)
        
        # Process several requests
        for i in range(5):
            request = APIRequest(
                request_id=f"req_{i}",
                tenant_id=f"t{i%2}",
                user_id="user_1",
                endpoint="test",
            )
            self.gateway.process_request(request)
        
        metrics = self.gateway._metrics.get_metrics()
        self.assertEqual(metrics["total_requests"], 5)
        self.assertGreater(metrics["avg_response_time_ms"], 0)


class TestUsageTracking(unittest.TestCase):
    """Test usage tracking and metering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = UsageEventLogger()
        self.metering = MeteringEngine()
    
    def test_log_usage_event(self):
        """Test logging usage event."""
        event = UsageEvent(
            event_id="evt_1",
            tenant_id="t1",
            user_id="user_1",
            event_type=UsageEventType.VIDEO_GENERATION,
            duration_seconds=30.0,
            resource_consumed=120.0,
            cost=5.0,
        )
        
        event_id = self.logger.log_event(event)
        self.assertEqual(event_id, "evt_1")
    
    def test_get_events_by_type(self):
        """Test retrieving events by type."""
        for i in range(3):
            event = UsageEvent(
                event_id=f"evt_{i}",
                tenant_id="t1",
                user_id="user_1",
                event_type=UsageEventType.API_CALL,
            )
            self.logger.log_event(event)
        
        events = self.logger.get_events("t1", event_type=UsageEventType.API_CALL)
        self.assertEqual(len(events), 3)
    
    def test_record_usage_quotas(self):
        """Test recording usage against quotas."""
        # Record some usage
        current = self.metering.record_usage("t1", "api_calls", 50.0)
        self.assertEqual(current, 50.0)
        
        # Check quota
        within_quota, current = self.metering.check_quota("t1", "api_calls", 100.0)
        self.assertTrue(within_quota)
        
        # Exceed quota
        self.metering.record_usage("t1", "api_calls", 60.0)
        within_quota, current = self.metering.check_quota("t1", "api_calls", 100.0)
        self.assertFalse(within_quota)
    
    def test_usage_reset(self):
        """Test usage reset."""
        self.metering.record_usage("t1", "api_calls", 50.0)
        self.metering.reset_usage("t1", "api_calls")
        
        current = self.metering.get_current_usage("t1", "api_calls")
        self.assertEqual(current, 0.0)
    
    def test_aggregate_usage(self):
        """Test aggregating usage metrics."""
        # Log some events
        for i in range(3):
            event = UsageEvent(
                event_id=f"evt_{i}",
                tenant_id="t1",
                user_id="user_1",
                event_type=UsageEventType.VIDEO_GENERATION,
                resource_consumed=30.0,
                cost=1.5,
            )
            self.logger.log_event(event)
        
        # Aggregate
        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow()
        metrics = self.logger.aggregate_usage("t1", start, end)
        
        self.assertEqual(metrics.tenant_id, "t1")


if __name__ == "__main__":
    unittest.main()

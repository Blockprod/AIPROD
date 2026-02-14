"""
API Gateway and Rate Limiting for Multi-Tenant SaaS.

Provides request routing, authentication, rate limiting,
and input validation.

Core Classes:
  - RateLimiter: Rate limiting by tenant/user
  - APIGateway: Request routing and processing
  - RequestValidator: Input validation
  - GatewayMetrics: Gateway performance tracking
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, Callable, List
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict
import time


class RateLimitWindow(str, Enum):
    """Rate limit window types."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_window: int
    window_type: RateLimitWindow
    burst_size: int = 0  # Allow temporary burst
    
    def get_window_seconds(self) -> int:
        """Get window size in seconds."""
        if self.window_type == RateLimitWindow.MINUTE:
            return 60
        elif self.window_type == RateLimitWindow.HOUR:
            return 3600
        elif self.window_type == RateLimitWindow.DAY:
            return 86400
        return 60


@dataclass
class RequestQuota:
    """Quota tracking for a request source."""
    requests_made: int = 0
    last_reset: datetime = field(default_factory=datetime.utcnow)
    burst_tokens: int = 0


@dataclass
class APIRequest:
    """API request wrapper."""
    request_id: str
    tenant_id: str
    user_id: str
    api_key: Optional[str] = None
    endpoint: str = ""
    method: str = "POST"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    headers: Dict[str, str] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""


@dataclass
class APIResponse:
    """API response wrapper."""
    request_id: str
    status_code: int
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "status_code": self.status_code,
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "processing_time_ms": self.processing_time_ms,
        }


class RateLimiter:
    """Token bucket rate limiter per tenant/user."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self._quotas: Dict[str, Dict[str, RequestQuota]] = defaultdict(dict)  # tenant -> user -> quota
        self._tenant_limits: Dict[str, RateLimitConfig] = {}  # tenant -> config
        self._default_config = RateLimitConfig(
            requests_per_window=1000,
            window_type=RateLimitWindow.HOUR,
        )
        self._lock = threading.RLock()
    
    def set_tenant_limit(self, tenant_id: str, config: RateLimitConfig) -> None:
        """Set rate limit for tenant."""
        with self._lock:
            self._tenant_limits[tenant_id] = config
    
    def check_rate_limit(self, tenant_id: str, user_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit."""
        with self._lock:
            config = self._tenant_limits.get(tenant_id, self._default_config)
            key = f"{tenant_id}:{user_id}"
            
            if key not in self._quotas[tenant_id]:
                self._quotas[tenant_id][key] = RequestQuota(burst_tokens=config.burst_size)
            
            quota = self._quotas[tenant_id][key]
            now = datetime.utcnow()
            
            # Check if we need to reset
            window_seconds = config.get_window_seconds()
            elapsed = (now - quota.last_reset).total_seconds()
            
            if elapsed >= window_seconds:
                quota.requests_made = 0
                quota.last_reset = now
                quota.burst_tokens = config.burst_size
            
            # Add burst token if available
            if quota.burst_tokens > 0 and quota.requests_made >= config.requests_per_window:
                quota.burst_tokens -= 1
                quota.requests_made += 1
                
                return True, {
                    "allowed": True,
                    "burst_used": True,
                    "remaining_requests": 0,
                    "remaining_burst": quota.burst_tokens,
                }
            
            # Normal rate limit check
            if quota.requests_made < config.requests_per_window:
                quota.requests_made += 1
                remaining = config.requests_per_window - quota.requests_made
                
                return True, {
                    "allowed": True,
                    "remaining_requests": remaining,
                    "reset_time": (quota.last_reset + timedelta(seconds=window_seconds)).isoformat(),
                }
            
            return False, {
                "allowed": False,
                "remaining_requests": 0,
                "reset_time": (quota.last_reset + timedelta(seconds=window_seconds)).isoformat(),
            }


class RequestValidator:
    """Validates API requests."""
    
    def __init__(self):
        """Initialize validator."""
        self._validators: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_validator(self, endpoint: str, validator: Callable) -> None:
        """Register validator for endpoint."""
        self._validators[endpoint].append(validator)
    
    def validate_request(self, request: APIRequest) -> Tuple[bool, List[str]]:
        """Validate request."""
        errors: List[str] = []
        
        # Basic validation
        if not request.tenant_id:
            errors.append("tenant_id is required")
        if not request.user_id:
            errors.append("user_id is required")
        if not request.endpoint:
            errors.append("endpoint is required")
        
        # Run registered validators
        validators = self._validators.get(request.endpoint, [])
        for validator in validators:
            try:
                is_valid, message = validator(request)
                if not is_valid:
                    errors.append(message)
            except Exception as e:
                errors.append(f"Validator error: {str(e)}")
        
        return len(errors) == 0, errors


class APIGateway:
    """Main API gateway for request handling."""
    
    def __init__(self):
        """Initialize API gateway."""
        self.rate_limiter = RateLimiter()
        self.validator = RequestValidator()
        self._request_handlers: Dict[str, Callable] = {}
        self._middleware: List[Callable] = []
        self._metrics = GatewayMetrics()
        self._lock = threading.RLock()
    
    def register_handler(self, endpoint: str, handler: Callable) -> None:
        """Register request handler for endpoint."""
        with self._lock:
            self._request_handlers[endpoint] = handler
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to request pipeline."""
        with self._lock:
            self._middleware.append(middleware)
    
    def process_request(self, request: APIRequest) -> APIResponse:
        """Process API request through gateway."""
        start_time = time.time()
        request_id = request.request_id or f"req_{int(time.time() * 1000)}"
        
        response = APIResponse(
            request_id=request_id,
            status_code=500,
            success=False,
            message="Unknown error",
        )
        
        try:
            # 1. Rate limiting check
            allowed, rate_info = self.rate_limiter.check_rate_limit(
                request.tenant_id,
                request.user_id,
            )
            if not allowed:
                response.status_code = 429
                response.message = "Rate limit exceeded"
                response.data = rate_info
                self._metrics.record_request(request.tenant_id, 429, start_time)
                return response
            
            # 2. Request validation
            is_valid, validation_errors = self.validator.validate_request(request)
            if not is_valid:
                response.status_code = 400
                response.message = "Request validation failed"
                response.data = {"errors": validation_errors}
                self._metrics.record_request(request.tenant_id, 400, start_time)
                return response
            
            # 3. Run middleware
            for middleware in self._middleware:
                middleware_response = middleware(request)
                if middleware_response:
                    response.status_code = middleware_response.get("status_code", 403)
                    response.message = middleware_response.get("message", "Middleware rejected request")
                    self._metrics.record_request(request.tenant_id, response.status_code, start_time)
                    return response
            
            # 4. Route to handler
            with self._lock:
                handler = self._request_handlers.get(request.endpoint)
            
            if not handler:
                response.status_code = 404
                response.message = f"No handler for endpoint: {request.endpoint}"
                self._metrics.record_request(request.tenant_id, 404, start_time)
                return response
            
            # Execute handler
            handler_response = handler(request)
            response.status_code = handler_response.get("status_code", 200)
            response.success = response.status_code < 400
            response.message = handler_response.get("message", "Success" if response.success else "Error")
            response.data = handler_response.get("data", {})
            
        except Exception as e:
            response.status_code = 500
            response.message = f"Internal server error: {str(e)}"
            response.success = False
        
        finally:
            response.processing_time_ms = (time.time() - start_time) * 1000
            self._metrics.record_request(request.tenant_id, response.status_code, start_time)
        
        return response


class GatewayMetrics:
    """Tracks gateway performance metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._status_counts: Dict[int, int] = defaultdict(int)
        self._response_times: List[float] = []
        self._lock = threading.RLock()
    
    def record_request(
        self,
        tenant_id: str,
        status_code: int,
        start_time: float,
    ) -> None:
        """Record request metrics."""
        processing_time = (time.time() - start_time) * 1000
        
        with self._lock:
            self._request_counts[tenant_id] += 1
            self._status_counts[status_code] += 1
            self._response_times.append(processing_time)
            
            # Keep only last 10000 response times
            if len(self._response_times) > 10000:
                self._response_times.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        with self._lock:
            response_times = self._response_times
            request_counts = dict(self._request_counts)
            status_counts = dict(self._status_counts)
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        return {
            "total_requests": sum(request_counts.values()),
            "total_tenants": len(request_counts),
            "requests_by_tenant": request_counts,
            "status_code_distribution": status_counts,
            "avg_response_time_ms": avg_response_time,
            "min_response_time_ms": min_response_time,
            "max_response_time_ms": max_response_time,
            "sample_size": len(response_times),
        }

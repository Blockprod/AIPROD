"""
Monitoring Middleware - Automatic metrics collection for all requests
"""

import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.datastructures import MutableHeaders
from src.monitoring.metrics_collector import get_metrics_collector
from src.utils.monitoring import logger


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware that automatically collects metrics for all requests.
    
    Tracks:
    - Request latency
    - Response size
    - Status codes
    - Errors
    - Per-endpoint statistics
    """
    
    # Paths to exclude from monitoring (health checks, etc)
    EXCLUDED_PATHS = {
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Track request metadata
        method = request.method
        path = request.url.path
        
        try:
            # Call next handler
            response = await call_next(request)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            status_code = response.status_code
            
            # Get response size
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body) if response.body else 0
            
            # Determine if error
            error_msg = None
            if status_code >= 400:
                error_msg = response.status_code
            
            # Record metrics
            collector = get_metrics_collector()
            collector.record_request(
                endpoint=path,
                method=method,
                duration_ms=duration_ms,
                status_code=status_code,
                response_size=response_size,
                error=error_msg
            )
            
            # Add metrics headers to response
            headers = MutableHeaders(response.headers)
            headers["X-Response-Time"] = str(round(duration_ms, 2))
            
            logger.debug(f"Recorded: {method} {path} {status_code} {duration_ms:.2f}ms")
            
            return response
        
        except Exception as e:
            # Record error
            duration_ms = (time.time() - start_time) * 1000
            collector = get_metrics_collector()
            collector.record_request(
                endpoint=path,
                method=method,
                duration_ms=duration_ms,
                status_code=500,
                error=str(e)
            )
            
            logger.error(f"Error in {method} {path}: {e}")
            
            # Re-raise to let normal error handling occur
            raise


class CacheMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that tracks cache hit/miss rates.
    Records cache performance for analytics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track cache metrics"""
        
        # Check if cache was used (custom header from cache middleware)
        response = await call_next(request)
        
        try:
            cache_hit = response.headers.get("X-Cache-Hit") == "true"
            collector = get_metrics_collector()
            collector.record_cache_hit(hit=cache_hit)
        except Exception as e:
            logger.warning(f"Error recording cache metric: {e}")
        
        return response


class ResourceMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that tracks resource usage.
    Could include memory, CPU, or external service calls.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track resource metrics"""
        
        # Track external service calls if header is present
        response = await call_next(request)
        
        try:
            # Check for external service timing headers
            db_time = response.headers.get("X-DB-Time")
            if db_time:
                try:
                    query_time = float(db_time)
                    collector = get_metrics_collector()
                    table = request.url.path.split('/')[-1]
                    collector.record_database_query(query_time, table)
                except (ValueError, IndexError):
                    pass
        
        except Exception as e:
            logger.warning(f"Error recording resource metric: {e}")
        
        return response

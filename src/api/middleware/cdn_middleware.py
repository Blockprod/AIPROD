"""
CDN Middleware for FastAPI
Applies appropriate Cache-Control headers based on content type
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.infra.cdn_config import get_cdn_config
from src.utils.monitoring import logger


class CDNHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply CDN-appropriate Cache-Control headers
    
    - Static assets: 1 year cache
    - HTML: 1 hour cache
    - API: 5 minute cache
    - Dynamic/Streaming: No cache
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add CDN headers to response"""
        response = await call_next(request)
        
        # Get CDN config
        cdn = get_cdn_config()
        
        # Get path from request
        path = request.url.path
        
        # Get cache headers for this path
        cache_headers = cdn.get_cache_headers(path)
        
        # Apply headers to response
        for header_name, header_value in cache_headers.items():
            response.headers[header_name] = header_value
        
        # Add CDN identifier header
        response.headers["X-CDN-Enabled"] = "true"
        
        # Log for monitoring
        logger.debug(f"CDN headers applied to {path}: {cache_headers.get('X-Cache-Policy', 'UNKNOWN')}")
        
        return response


class CDNMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track CDN metrics
    
    Records cache hit/miss rates for monitoring
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Track CDN metrics"""
        response = await call_next(request)
        
        # Check if served from cache (in real CDN, this would be from cache headers)
        # For now, we track in monitoring
        cdn = get_cdn_config()
        
        # Get response size
        content_length = response.headers.get("content-length", 0)
        try:
            bytes_served = int(content_length) if content_length else 0
        except ValueError:
            bytes_served = 0
        
        # Determine if would be cached (based on Cache-Control header)
        cache_control = response.headers.get("Cache-Control", "")
        would_be_cached = "no-cache" not in cache_control and "no-store" not in cache_control
        
        # Record request (approximate - real tracking would be via Cloud CDN metrics)
        cdn.monitoring.record_request(would_be_cached, bytes_served)
        
        return response

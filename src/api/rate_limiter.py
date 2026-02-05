"""
Rate limiting configuration and implementation for FastAPI.
Uses SlowAPI for flexible rate limiting per endpoint.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
from typing import Callable
import logging

logger = logging.getLogger(__name__)

# Initialize rate limiter with IP-based key function
limiter = Limiter(key_func=get_remote_address)


# Custom exception handler for rate limit exceeded
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Handle rate limit exceeded responses.
    Returns 429 with descriptive message.
    """
    client_host = request.client.host if request.client else "unknown"
    logger.warning(
        f"Rate limit exceeded for {client_host}: {exc.detail}"
    )
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded. Please retry after {exc.detail.split('in ')[-1] if 'in ' in exc.detail else '60 seconds'}"
    )


# Rate limit tiers by endpoint
RATE_LIMITS = {
    # Public endpoints - generous limits
    "health": "1000/minute",
    "docs": "100/minute",
    "openapi": "100/minute",
    
    # Core pipeline endpoints - moderate limits
    "pipeline_run": "30/minute",      # ~1 job per 2 seconds
    "pipeline_status": "60/minute",   # Status checks
    
    # Financial endpoints - moderate limits  
    "cost_estimate": "50/minute",
    "optimize": "20/minute",
    
    # Data endpoints - moderate limits
    "alerts": "60/minute",
    "icc_data": "60/minute",
    "financial_data": "60/minute",
    
    # QA endpoints - moderate limits
    "qa_technical": "40/minute",
    "qa_financial": "40/minute",
}


def get_rate_limit(endpoint_name: str) -> str:
    """
    Get rate limit for an endpoint.
    Falls back to default if endpoint not configured.
    
    Args:
        endpoint_name: Name of the endpoint (e.g., 'pipeline_run')
        
    Returns:
        Rate limit string (e.g., '30/minute')
    """
    return RATE_LIMITS.get(endpoint_name, "100/minute")  # Default: 100 req/min


def apply_rate_limit(endpoint_name: str) -> Callable:
    """
    Decorator to apply rate limiting to an endpoint.
    
    Args:
        endpoint_name: Name of the endpoint for rate limiting
        
    Returns:
        Decorated function with rate limiting
    """
    limit = get_rate_limit(endpoint_name)
    return limiter.limit(limit)

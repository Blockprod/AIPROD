"""
CDN Configuration for AIPROD
Configures Cloud CDN with intelligent caching policies
"""

from enum import Enum
from typing import Dict, Optional, Any
from datetime import timedelta
import json

class CachePolicy(Enum):
    """Cache policies by content type"""
    STATIC_ASSETS = {
        "ttl": 31536000,  # 1 year
        "extensions": [".js", ".css", ".png", ".jpg", ".gif", ".svg", ".woff2"],
        "immutable": True,
    }
    HTML_PAGES = {
        "ttl": 3600,  # 1 hour
        "extensions": [".html"],
        "must_revalidate": True,
    }
    API_RESPONSES = {
        "ttl": 300,  # 5 minutes
        "patterns": ["/api/", "/v1/"],
        "must_revalidate": True,
    }
    DYNAMIC_CONTENT = {
        "ttl": 0,  # No cache
        "patterns": ["/pipeline/run", "/pipeline/*/result"],
        "no_cache": True,
    }


class CDNConfig:
    """
    Cloud CDN Configuration
    
    Implements intelligent caching strategy with:
    - Time-based cache invalidation
    - Content-type based policies
    - Geographic awareness
    - Cache hit rate monitoring
    """
    
    def __init__(self):
        """Initialize CDN configuration"""
        self.policies = CachePolicy.__members__
        self.cache_directives = self._build_cache_directives()
        self.monitoring = CDNMonitoring()
        
    def _build_cache_directives(self) -> Dict[str, str]:
        """Build Cache-Control directives for each policy"""
        directives = {}
        
        directives["static"] = "public, max-age=31536000, immutable"
        directives["html"] = "public, max-age=3600, must-revalidate"
        directives["api"] = "public, max-age=300, must-revalidate"
        directives["dynamic"] = "no-cache, no-store, must-revalidate, max-age=0"
        
        return directives
    
    def get_cache_headers(self, path: str) -> Dict[str, str]:
        """
        Get appropriate Cache-Control headers for a path
        
        Args:
            path: URL path
            
        Returns:
            Dict of headers to set
        """
        headers = {}
        
        # Dynamic content (highest priority - check first)
        if any(pattern in path for pattern in ["/pipeline/run", "/pipeline/", "/result", "/streaming"]):
            headers["Cache-Control"] = self.cache_directives["dynamic"]
            headers["X-Cache-Policy"] = "DYNAMIC"
            headers["Pragma"] = "no-cache"
            return headers
        
        # Static assets
        if any(path.endswith(ext) for ext in [".js", ".css", ".png", ".jpg", ".gif", ".svg", ".woff2"]):
            headers["Cache-Control"] = self.cache_directives["static"]
            headers["X-Cache-Policy"] = "STATIC"
            return headers
            
        # HTML pages
        if path.endswith(".html") or path == "/":
            headers["Cache-Control"] = self.cache_directives["html"]
            headers["X-Cache-Policy"] = "HTML"
            return headers
        
        # API responses (check for patterns)
        if "/api/" in path or "/v1/" in path:
            headers["Cache-Control"] = self.cache_directives["api"]
            headers["X-Cache-Policy"] = "API"
            return headers
        
        # Unknown - be conservative
        headers["Cache-Control"] = self.cache_directives["dynamic"]
        headers["X-Cache-Policy"] = "UNKNOWN"
        return headers
    
    def get_cdn_configuration(self) -> Dict[str, Any]:
        """
        Get Cloud CDN configuration for terraform/gcloud
        
        Returns:
            Dict with CDN configuration
        """
        return {
            "enable_cdn": True,
            "cache_mode": "CACHE_ALL_STATIC",
            "client_ttl": 3600,
            "default_ttl": 3600,
            "max_ttl": 31536000,
            "negative_caching": True,
            "negative_caching_policy": [
                {"code": 404, "ttl": 120},
                {"code": 410, "ttl": 120},
                {"code": 501, "ttl": 60},
            ],
            "serve_while_stale": 86400,
            "custom_request_headers": {
                "headers": ["User-Agent: AIPROD-V33"]
            }
        }


class CDNMonitoring:
    """Monitor CDN performance and cache hit rates"""
    
    def __init__(self):
        """Initialize CDN monitoring"""
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "bytes_served_from_cache": 0,
            "bytes_served_from_origin": 0,
        }
    
    def record_request(self, from_cache: bool, bytes_served: int):
        """Record a CDN request"""
        self.metrics["total_requests"] += 1
        
        if from_cache:
            self.metrics["cache_hits"] += 1
            self.metrics["bytes_served_from_cache"] += bytes_served
        else:
            self.metrics["cache_misses"] += 1
            self.metrics["bytes_served_from_origin"] += bytes_served
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        if self.metrics["total_requests"] == 0:
            return 0.0
        return (self.metrics["cache_hits"] / self.metrics["total_requests"]) * 100
    
    def get_bandwidth_saved(self) -> float:
        """Calculate bandwidth saved by caching"""
        if self.metrics["bytes_served_from_origin"] == 0:
            return 0.0
        total = self.metrics["bytes_served_from_cache"] + self.metrics["bytes_served_from_origin"]
        return (self.metrics["bytes_served_from_cache"] / total) * 100 if total > 0 else 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of CDN metrics"""
        return {
            "total_requests": self.metrics["total_requests"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate_percent": round(self.get_cache_hit_rate(), 2),
            "bandwidth_from_cache_percent": round(self.get_bandwidth_saved(), 2),
            "bytes_served_from_cache": self.metrics["bytes_served_from_cache"],
            "bytes_served_from_origin": self.metrics["bytes_served_from_origin"],
        }


# Global CDN instance
_cdn_config = None

def get_cdn_config() -> CDNConfig:
    """Get global CDN configuration instance"""
    global _cdn_config
    if _cdn_config is None:
        _cdn_config = CDNConfig()
    return _cdn_config

"""
Compression Middleware - Response compression for faster delivery
Supports gzip and deflate compression
"""

import gzip
from io import BytesIO
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse
from starlette.datastructures import MutableHeaders
from src.utils.monitoring import logger


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for compressing responses.
    
    Features:
    - Gzip compression support
    - Deflate compression support
    - Automatic threshold-based compression
    - Accept-Encoding header respecting
    - Compression ratio tracking
    """
    
    # Minimum size to compress (bytes)
    MIN_COMPRESSION_SIZE = 500
    
    # Content types that benefit from compression
    COMPRESSIBLE_TYPES = {
        "application/json",
        "application/javascript",
        "text/plain",
        "text/html",
        "text/css",
        "text/xml",
        "application/xml",
    }
    
    def __init__(self, app, minimum_size: int = 500):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_stats = {
            "compressed_count": 0,
            "skipped_count": 0,
            "total_before": 0,
            "total_after": 0,
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and compress response if applicable"""
        
        response = await call_next(request)
        
        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "").lower()
        if "gzip" not in accept_encoding and "deflate" not in accept_encoding:
            return response
        
        # Check response content type
        content_type = response.headers.get("content-type", "").lower()
        if not any(ct in content_type for ct in self.COMPRESSIBLE_TYPES):
            return response
        
        # Try to compress the response
        if hasattr(response, "body"):
            try:
                body = response.body
                
                # Check minimum size
                if len(body) < self.minimum_size:
                    self.compression_stats["skipped_count"] += 1
                    return response
                
                # Compress
                if "gzip" in accept_encoding:
                    compressed = self._gzip_compress(body)
                    encoding = "gzip"
                elif "deflate" in accept_encoding:
                    compressed = self._deflate_compress(body)
                    encoding = "deflate"
                else:
                    return response
                
                # Only use compression if it actually reduces size
                if len(compressed) < len(body):
                    headers = MutableHeaders(response.headers)
                    headers["Content-Encoding"] = encoding
                    headers["Content-Length"] = str(len(compressed))
                    
                    # Track stats
                    self.compression_stats["compressed_count"] += 1
                    self.compression_stats["total_before"] += len(body)
                    self.compression_stats["total_after"] += len(compressed)
                    
                    ratio = round((len(compressed) / len(body)) * 100, 1)
                    logger.debug(f"Compressed response: {ratio}% of original ({len(body)}→{len(compressed)} bytes)")
                    
                    return Response(
                        content=compressed,
                        status_code=response.status_code,
                        headers=dict(headers),
                        media_type=response.media_type,
                    )
            
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
                return response
        
        return response
    
    @staticmethod
    def _gzip_compress(data: bytes) -> bytes:
        """Compress data with gzip"""
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
            f.write(data)
        return buf.getvalue()
    
    @staticmethod
    def _deflate_compress(data: bytes) -> bytes:
        """Compress data with deflate"""
        import zlib
        return zlib.compress(data, 6)
    
    def get_stats(self):
        """Get compression statistics"""
        if self.compression_stats["total_before"] == 0:
            compression_ratio = 0
        else:
            compression_ratio = round(
                (self.compression_stats["total_after"] / 
                 self.compression_stats["total_before"]) * 100,
                2
            )
        
        return {
            "compressed_responses": self.compression_stats["compressed_count"],
            "skipped_responses": self.compression_stats["skipped_count"],
            "total_bytes_before": self.compression_stats["total_before"],
            "total_bytes_after": self.compression_stats["total_after"],
            "compression_ratio": compression_ratio,
            "bytes_saved": self.compression_stats["total_before"] - self.compression_stats["total_after"],
        }


class CacheHeaderMiddleware(BaseHTTPMiddleware):
    """
    Middleware for setting appropriate cache headers.
    Improves browser caching and CDN efficiency.
    """
    
    # Cache control patterns
    CACHE_PATTERNS = {
        "/static": "public, max-age=31536000, immutable",  # 1 year for static
        "/api": "private, max-age=300",  # 5 minutes for API
        "/monitoring": "private, max-age=60",  # 1 minute for monitoring
        "/health": "no-cache, no-store, must-revalidate",  # No cache for health
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add cache headers to response"""
        
        response = await call_next(request)
        
        # Determine cache header based on path
        cache_header = "private, max-age=300"  # Default
        
        for pattern, header in self.CACHE_PATTERNS.items():
            if request.url.path.startswith(pattern):
                cache_header = header
                break
        
        headers = MutableHeaders(response.headers)
        headers["Cache-Control"] = cache_header
        
        if "no-cache" not in cache_header:
            # Add ETag for cacheable responses
            headers["ETag"] = f'"{hash(response.body if hasattr(response, "body") else "")}"'
        
        return response

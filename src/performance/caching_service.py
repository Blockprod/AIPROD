"""
Caching Service - Multi-backend caching with Redis support
Provides request-level, endpoint-level, and query caching
"""

import json
import time
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from hashlib import md5
from src.utils.monitoring import logger


class CacheBackend(str, Enum):
    """Available cache backends"""
    MEMORY = "memory"
    REDIS = "redis"


class CacheEntry:
    """Individual cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl_seconds: int = 300):
        self.key = key
        self.value = value
        self.ttl_seconds = ttl_seconds
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds == 0:
            return False  # No expiration
        
        age = time.time() - self.created_at
        return age > self.ttl_seconds
    
    def access(self) -> Any:
        """Record access and return value"""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache entry statistics"""
        return {
            "key": self.key,
            "ttl_seconds": self.ttl_seconds,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_accessed": datetime.fromtimestamp(self.last_accessed).isoformat(),
            "access_count": self.access_count,
            "expired": self.is_expired(),
        }


class InMemoryCache:
    """In-memory cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, cleanup_interval: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            return None
        
        return entry.access()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache"""
        # Cleanup if needed
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
        
        # Evict LRU if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = CacheEntry(key, value, ttl_seconds)
    
    def delete(self, key: str):
        """Delete entry from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        expired_count = sum(1 for e in self.cache.values() if e.is_expired())
        
        # Estimate memory usage approximately (1KB per entry)
        memory_estimate = (len(self.cache) * 1024) / 1024 / 1024
        
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "memory_estimate_mb": memory_estimate,
        }
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [
            k for k, v in self.cache.items()
            if v.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        self.last_cleanup = time.time()
        logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[lru_key]
        logger.debug(f"Cache eviction: removed LRU entry {lru_key}")


class CachingService:
    """
    High-level caching service with multiple backends.
    
    Features:
    - Request-level caching (entire response)
    - Query-level caching (database results)
    - Endpoint-level caching (common patterns)
    - Cache invalidation strategies
    - Hit/miss tracking
    - Intelligent TTL management
    """
    
    def __init__(self, backend: CacheBackend = CacheBackend.MEMORY):
        self.backend = backend
        self.memory_cache = InMemoryCache(max_size=5000)
        self.hit_count = 0
        self.miss_count = 0
    
    def generate_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate cache key from namespace and arguments"""
        key_parts = [namespace]
        
        # Add positional args
        for arg in args:
            key_parts.append(str(arg))
        
        # Add kwargs in sorted order
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")
        
        key_string = ":".join(key_parts)
        return md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.memory_cache.get(key)
        
        if value is not None:
            self.hit_count += 1
            logger.debug(f"Cache HIT: {key}")
            return value
        
        self.miss_count += 1
        logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache"""
        self.memory_cache.set(key, value, ttl_seconds)
        logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
    
    def delete(self, key: str):
        """Delete key from cache"""
        self.memory_cache.delete(key)
        logger.debug(f"Cache DELETE: {key}")
    
    def delete_by_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        keys_to_delete = [
            k for k in self.memory_cache.cache.keys()
            if pattern in k
        ]
        
        for key in keys_to_delete:
            self.delete(key)
        
        logger.debug(f"Cache INVALIDATE: {len(keys_to_delete)} entries matching '{pattern}'")
    
    def cache_request(self, endpoint: str, method: str, ttl_seconds: int = 300):
        """Decorator for request-level caching"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = self.generate_key(f"request:{method}:{endpoint}", *args, **kwargs)
                
                # Try to get from cache
                cached = self.get(key)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl_seconds)
                
                return result
            
            return wrapper
        return decorator
    
    def cache_query(self, ttl_seconds: int = 600):
        """Decorator for query-level caching"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key from query
                key = self.generate_key("query", *args, **kwargs)
                
                # Try to get from cache
                cached = self.get(key)
                if cached is not None:
                    return cached
                
                # Execute query
                result = await func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl_seconds)
                
                return result
            
            return wrapper
        return decorator
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        
        return (self.hit_count / total) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "backend": self.backend.value,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.get_hit_rate(), 2),
            "cache_stats": self.memory_cache.get_stats(),
        }
    
    def reset_stats(self):
        """Reset hit/miss counters"""
        self.hit_count = 0
        self.miss_count = 0


# Global caching service instance
_caching_service = None


def get_caching_service(backend: CacheBackend = CacheBackend.MEMORY) -> CachingService:
    """Get or create singleton caching service"""
    global _caching_service
    if _caching_service is None:
        _caching_service = CachingService(backend)
    return _caching_service

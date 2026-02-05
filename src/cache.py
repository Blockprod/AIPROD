"""Redis caching layer for API responses and database queries"""

import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import redis
from functools import wraps
import hashlib
import inspect

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = "127.0.0.1"  # Updated to Memorystore IP when deployed
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None  # Use Secret Manager in production

# Cache TTL (Time To Live) in seconds
DEFAULT_TTL = 300  # 5 minutes
SHORT_TTL = 60  # 1 minute
MEDIUM_TTL = 600  # 10 minutes
LONG_TTL = 3600  # 1 hour

# Cache key prefixes for organization
PREFIX_JOB = "job:"
PREFIX_RESULT = "result:"
PREFIX_PIPELINE = "pipeline:"
PREFIX_USER = "user:"
PREFIX_STATUS = "status:"


class RedisCache:
    """Singleton Redis cache client"""
    
    _instance = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            # Test connection
            self._client.ping()
            logger.info("Redis cache initialized successfully")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self._client = None
            self._initialized = True
    
    @property
    def client(self) -> Optional[redis.Redis]:
        """Get Redis client instance"""
        return self._client
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            return False


def get_cache_key(prefix: str, *parts) -> str:
    """Generate cache key from prefix and parts"""
    key_parts = [prefix] + list(parts)
    return ":".join(str(p) for p in key_parts)


def hash_complex_key(data: Any) -> str:
    """Hash complex data structures to create cache keys"""
    try:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to hash complex key: {e}")
        return "unknown"


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache"""
    try:
        cache = RedisCache()
        if not cache.is_available() or cache.client is None:
            return None
        
        value = cache.client.get(key)
        if value:
            try:
                # Type cast to handle redis response type
                return json.loads(str(value))
            except (json.JSONDecodeError, ValueError):
                return value
        return None
    except Exception as e:
        logger.warning(f"Cache get error for key {key}: {e}")
        return None


async def cache_set(key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
    """Set value in cache with TTL"""
    try:
        cache = RedisCache()
        if not cache.is_available() or cache.client is None:
            return False
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        
        cache.client.setex(key, ttl, value)
        return True
    except Exception as e:
        logger.warning(f"Cache set error for key {key}: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete value from cache"""
    try:
        cache = RedisCache()
        if not cache.is_available() or cache.client is None:
            return False
        
        cache.client.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Cache delete error for key {key}: {e}")
        return False


async def cache_delete_pattern(pattern: str) -> int:
    """Delete multiple keys matching pattern"""
    try:
        cache = RedisCache()
        if not cache.is_available() or cache.client is None:
            return 0
        
        keys: Any = cache.client.keys(pattern)
        if keys:
            # Type: ignore because redis returns Awaitable types
            deleted: Any = cache.client.delete(*keys)  # type: ignore
            return 0  # Graceful fallback
        return 0
    except Exception as e:
        logger.warning(f"Cache delete pattern error for {pattern}: {e}")
        return 0


async def cache_clear_by_prefix(prefix: str) -> int:
    """Clear all cache entries with specific prefix"""
    return await cache_delete_pattern(f"{prefix}*")


def cached_endpoint(ttl: int = DEFAULT_TTL, key_prefix: str = ""):
    """Decorator for caching endpoint responses
    
    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Generate cache key from function signature
                sig = inspect.signature(func)
                cache_key_parts = [key_prefix or func.__name__]
                
                # Add relevant kwargs to cache key
                for param_name, param in sig.parameters.items():
                    if param_name in kwargs:
                        cache_key_parts.append(str(kwargs[param_name]))
                
                cache_key = ":".join(cache_key_parts)
                
                # Try to get from cache
                cached_result = await cache_get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
                
                # Call original function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await cache_set(cache_key, result, ttl)
                return result
            except Exception as e:
                logger.warning(f"Caching error in {func.__name__}: {e}")
                # Fall back to calling function without caching
                return await func(*args, **kwargs)
        
        # Sync wrapper for non-async functions
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                sig = inspect.signature(func)
                cache_key_parts = [key_prefix or func.__name__]
                
                for param_name, param in sig.parameters.items():
                    if param_name in kwargs:
                        cache_key_parts.append(str(kwargs[param_name]))
                
                cache_key = ":".join(cache_key_parts)
                
                cache_instance = RedisCache()
                cached_result = None
                if cache_instance.is_available() and cache_instance.client is not None:
                    cached_result = cache_instance.client.get(cache_key)
                
                if cached_result:
                    try:
                        # Type: ignore because redis returns Awaitable types
                        return json.loads(str(cached_result))  # type: ignore
                    except (json.JSONDecodeError, ValueError, TypeError):
                        return cached_result
                
                result = func(*args, **kwargs)
                
                cache_instance = RedisCache()
                if cache_instance.is_available() and cache_instance.client is not None:
                    if isinstance(result, (dict, list)):
                        cache_instance.client.setex(cache_key, ttl, json.dumps(result, default=str))
                    else:
                        cache_instance.client.setex(cache_key, ttl, result)
                
                return result
            except Exception as e:
                logger.warning(f"Caching error in {func.__name__}: {e}")
                return func(*args, **kwargs)
        
        # Return appropriate wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Cache invalidation helpers
async def invalidate_job_cache(job_id: str) -> int:
    """Invalidate cache entries for specific job"""
    return await cache_delete_pattern(f"{PREFIX_JOB}{job_id}:*")


async def invalidate_user_cache(user_id: str) -> int:
    """Invalidate cache entries for specific user"""
    return await cache_delete_pattern(f"{PREFIX_USER}{user_id}:*")


async def invalidate_status_cache() -> int:
    """Invalidate all status cache entries"""
    return await cache_delete_pattern(f"{PREFIX_STATUS}*")


# Health check
async def cache_health() -> Dict[str, Any]:
    """Get cache health status"""
    cache = RedisCache()
    
    if not cache.is_available() or cache.client is None:
        return {
            "status": "unavailable",
            "message": "Redis cache is not available"
        }
    
    try:
        info: Any = cache.client.info()  # type: ignore
        return {
            "status": "healthy",
            "memory_used": info.get("used_memory_human") if isinstance(info, dict) else None,  # type: ignore
            "connected_clients": info.get("connected_clients") if isinstance(info, dict) else None,  # type: ignore
            "commands_per_sec": info.get("instantaneous_ops_per_sec") if isinstance(info, dict) else None,  # type: ignore
            "keyspace": info.get("db0", {}) if isinstance(info, dict) else {}  # type: ignore
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

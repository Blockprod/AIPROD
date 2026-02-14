"""
Batch Sizing Cache and Index

Caches previously computed optimal batch sizes keyed by model, device, and constraints.
Enables fast lookup of pre-optimized batch sizes without repeated profiling.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import hashlib
import json


@dataclass
class CacheKey:
    """Composite key for caching batch sizing decisions."""
    model_name: str
    model_hash: str  # Hash of model architecture
    device_type: str
    device_compute_capability: str  # e.g., "8.9" for A100
    memory_available_mb: int
    target_optimization: str  # "throughput", "memory", "power", "balanced"
    
    def to_string(self) -> str:
        """Convert key to hashable string."""
        components = [
            self.model_name,
            self.model_hash[:8],  # First 8 chars of hash
            self.device_type,
            self.device_compute_capability,
            str(self.memory_available_mb),
            self.target_optimization,
        ]
        return "|".join(components)
    
    def fuzzy_match(self, other: "CacheKey", tolerance: float = 0.1) -> bool:
        """Check if keys match within tolerance (e.g., for memory budget)."""
        if (self.model_name != other.model_name or
            self.device_type != other.device_type):
            return False
        
        # Allow 10% memory variance
        memory_diff = abs(self.memory_available_mb - other.memory_available_mb)
        max_memory = max(self.memory_available_mb, other.memory_available_mb)
        return memory_diff / max_memory <= tolerance


@dataclass
class CacheEntry:
    """Single entry in batch size cache."""
    key: CacheKey
    batch_size: int
    predicted_latency_ms: float
    predicted_memory_mb: float
    predicted_throughput: float
    confidence_score: float
    timestamp: float
    hits: int = 0  # Number of times this entry was used
    
    def is_stale(self, max_age_seconds: float = 86400.0) -> bool:
        """Check if cache entry is too old."""
        import time
        return (time.time() - self.timestamp) > max_age_seconds


class BatchSizingCache:
    """LRU cache for batch sizing decisions."""
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: Dict[str, int] = {}  # For LRU tracking
        self.access_counter = 0
        
    def put(self, entry: CacheEntry) -> None:
        """Add or update cache entry."""
        key_str = entry.key.to_string()
        
        # If cache full, evict LRU item
        if len(self.cache) >= self.max_entries and key_str not in self.cache:
            lru_key = min(self.access_order, key=self.access_order.get)
            del self.cache[lru_key]
            del self.access_order[lru_key]
        
        self.cache[key_str] = entry
        self.access_counter += 1
        self.access_order[key_str] = self.access_counter
    
    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Retrieve cache entry."""
        key_str = key.to_string()
        
        if key_str in self.cache:
            entry = self.cache[key_str]
            self.access_counter += 1
            self.access_order[key_str] = self.access_counter
            entry.hits += 1
            
            if not entry.is_stale():
                return entry
            else:
                # Remove stale entry
                del self.cache[key_str]
        
        return None
    
    def fuzzy_get(self, key: CacheKey, tolerance: float = 0.1) -> Optional[CacheEntry]:
        """Retrieve cache entry with fuzzy matching."""
        best_match = None
        best_confidence = 0.0
        
        for cached_entry in self.cache.values():
            if key.fuzzy_match(cached_entry.key, tolerance):
                if cached_entry.confidence_score > best_confidence:
                    best_match = cached_entry
                    best_confidence = cached_entry.confidence_score
        
        return best_match
    
    def invalidate(self, model_name: Optional[str] = None, device_type: Optional[str] = None) -> int:
        """Invalidate cache entries matching criteria."""
        keys_to_remove = []
        
        for key_str, entry in self.cache.items():
            if model_name and entry.key.model_name != model_name:
                continue
            if device_type and entry.key.device_type != device_type:
                continue
            keys_to_remove.append(key_str)
        
        for key_str in keys_to_remove:
            del self.cache[key_str]
            del self.access_order[key_str]
        
        return len(keys_to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"entries": 0, "hit_rate": 0.0}
        
        total_hits = sum(e.hits for e in self.cache.values())
        avg_confidence = sum(e.confidence_score for e in self.cache.values()) / len(self.cache)
        
        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "total_hits": total_hits,
            "avg_confidence": avg_confidence,
            "hit_rate": total_hits / len(self.cache) if self.cache else 0.0,
        }


class ModelProfileRegistry:
    """Registry of model performance profiles."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        
    def register_profile(
        self,
        model_name: str,
        model_hash: str,
        profile_data: Dict[str, Any],
    ) -> None:
        """Register model performance profile."""
        key = f"{model_name}:{model_hash[:8]}"
        self.profiles[key] = {
            "model_name": model_name,
            "model_hash": model_hash,
            "profile_data": profile_data,
            "timestamp": __import__("time").time(),
        }
    
    def get_profile(self, model_name: str, model_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve model profile."""
        key = f"{model_name}:{model_hash[:8]}"
        return self.profiles.get(key, {}).get("profile_data")
    
    def export_profiles(self) -> str:
        """Export profiles as JSON for persistence."""
        return json.dumps(self.profiles, indent=2)
    
    def import_profiles(self, json_data: str) -> None:
        """Import profiles from JSON."""
        self.profiles = json.loads(json_data)
    
    def get_similar_profiles(self, model_name: str) -> Dict[str, Any]:
        """Get profiles for similar model names."""
        return {
            k: v for k, v in self.profiles.items()
            if model_name.lower() in k.lower()
        }


class BatchSizeRecommender:
    """Recommends batch sizes based on cache and profiling."""
    
    def __init__(self, cache: BatchSizingCache, registry: ModelProfileRegistry):
        self.cache = cache
        self.registry = registry
    
    def recommend(
        self,
        model_name: str,
        model_hash: str,
        device_type: str,
        device_compute: str,
        memory_mb: int,
        optimization_target: str = "balanced",
        use_fuzzy: bool = True,
    ) -> Tuple[int, float]:
        """Recommend batch size and confidence score."""
        key = CacheKey(
            model_name=model_name,
            model_hash=model_hash,
            device_type=device_type,
            device_compute_capability=device_compute,
            memory_available_mb=memory_mb,
            target_optimization=optimization_target,
        )
        
        # Try exact match
        entry = self.cache.get(key)
        if entry:
            return entry.batch_size, entry.confidence_score
        
        # Try fuzzy match
        if use_fuzzy:
            entry = self.cache.fuzzy_get(key)
            if entry:
                return entry.batch_size, entry.confidence_score * 0.8  # Lower confidence for fuzzy
        
        # Consult profile registry
        profile = self.registry.get_profile(model_name, model_hash)
        if profile:
            # Estimate from profile metadata
            default_batch = 32
            return default_batch, 0.6
        
        # Default fallback
        return 32, 0.3

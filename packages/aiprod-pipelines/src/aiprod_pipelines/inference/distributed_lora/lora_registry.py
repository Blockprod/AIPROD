"""
LoRA Model Registry and Catalog

Manages a global registry of all LoRA models with versioning,
discovery, and lifecycle management.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


class ModelStatus(Enum):
    """Status of LoRA model"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class AccessLevel(Enum):
    """Model access levels"""
    PRIVATE = "private"  # Only owner
    TENANT = "tenant"  # Tenant members
    PUBLIC = "public"  # Public catalog
    SHARED = "shared"  # Specific users


@dataclass
class ModelVersion:
    """Version of a LoRA model"""
    version_number: int
    created_at: datetime
    updated_at: datetime
    training_loss: float
    eval_metrics: Dict[str, float]
    checkpoint_path: str
    parent_version: Optional[int] = None
    changelog: str = ""


class LoRARegistry:
    """Global registry of LoRA models"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}  # model_id -> model_info
        self.model_index: Dict[str, List[str]] = {}  # index by tenant_id, user_id, tags
        self.versions: Dict[str, List[ModelVersion]] = {}  # model_id -> versions
        self.metrics_cache: Dict[str, Dict[str, float]] = {}
    
    def register_model(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Register new LoRA model"""
        if model_id in self.models:
            return False  # Already registered
        
        self.models[model_id] = {
            **metadata,
            "registered_at": datetime.utcnow().isoformat(),
            "status": ModelStatus.TRAINING.value,
            "access_level": AccessLevel.PRIVATE.value
        }
        
        # Index by tenant
        tenant_id = metadata.get("tenant_id")
        if tenant_id not in self.model_index:
            self.model_index[tenant_id] = []
        self.model_index[tenant_id].append(model_id)
        
        # Initialize versions list
        self.versions[model_id] = []
        
        return True
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        if model_id not in self.models:
            return False
        
        self.models[model_id]["status"] = status.value
        self.models[model_id]["last_updated"] = datetime.utcnow().isoformat()
        return True
    
    def add_version(self, model_id: str, version: ModelVersion) -> bool:
        """Add version to model"""
        if model_id not in self.versions:
            return False
        
        self.versions[model_id].append(version)
        self.models[model_id]["latest_version"] = version.version_number
        return True
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata"""
        return self.models.get(model_id)
    
    def get_model_version(self, model_id: str, version: int) -> Optional[ModelVersion]:
        """Get specific version of model"""
        if model_id not in self.versions:
            return None
        
        for v in self.versions[model_id]:
            if v.version_number == version:
                return v
        return None
    
    def list_models_by_tenant(self, tenant_id: str, 
                             status: Optional[ModelStatus] = None) -> List[str]:
        """List models by tenant"""
        models = self.model_index.get(tenant_id, [])
        
        if status:
            models = [m for m in models if self.models[m]["status"] == status.value]
        
        return models
    
    def list_models_by_user(self, user_id: str, tenant_id: str) -> List[str]:
        """List models owned by user"""
        models = []
        for model_id in self.model_index.get(tenant_id, []):
            if self.models[model_id].get("user_id") == user_id:
                models.append(model_id)
        return models
    
    def search_models(self, query: Dict[str, Any]) -> List[str]:
        """Search models by criteria"""
        results = []
        
        for model_id, model_info in self.models.items():
            match = True
            
            # Check tenant filter
            if "tenant_id" in query and model_info.get("tenant_id") != query["tenant_id"]:
                match = False
            
            # Check status filter
            if "status" in query and model_info.get("status") != query["status"]:
                match = False
            
            # Check tags
            if "tags" in query:
                model_tags = set(model_info.get("tags", []))
                query_tags = set(query["tags"])
                if not query_tags.issubset(model_tags):
                    match = False
            
            if match:
                results.append(model_id)
        
        return results
    
    def set_access_level(self, model_id: str, access_level: AccessLevel) -> bool:
        """Set model access level"""
        if model_id not in self.models:
            return False
        
        self.models[model_id]["access_level"] = access_level.value
        return True
    
    def share_model(self, model_id: str, user_ids: List[str]) -> bool:
        """Share model with specific users"""
        if model_id not in self.models:
            return False
        
        self.models[model_id]["shared_with"] = user_ids
        self.models[model_id]["access_level"] = AccessLevel.SHARED.value
        return True
    
    def cache_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Cache evaluation metrics"""
        self.metrics_cache[model_id] = {
            "metrics": metrics,
            "cached_at": datetime.utcnow().isoformat()
        }
    
    def get_cached_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Retrieve cached metrics"""
        if model_id in self.metrics_cache:
            return self.metrics_cache[model_id]["metrics"]
        return None
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        statuses = {}
        for model in self.models.values():
            status = model.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        
        return {
            "total_models": len(self.models),
            "total_versions": sum(len(v) for v in self.versions.values()),
            "status_breakdown": statuses,
            "num_tenants": len(self.model_index)
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry as dictionary"""
        return {
            "models": self.models,
            "versions": {k: [(v.version_number, v.training_loss) for v in vals] 
                        for k, vals in self.versions.items()},
            "stats": self.get_registry_stats()
        }


class ModelDiscovery:
    """Discovers and recommends LoRA models"""
    
    def __init__(self, registry: LoRARegistry):
        self.registry = registry
    
    def discover_similar_models(self, base_model: str, is_public: bool = True) -> List[str]:
        """Find models similar to given model"""
        base_info = self.registry.get_model(base_model)
        if not base_info:
            return []
        
        # Search for models with similar tags
        query = {"tags": base_info.get("tags", [])}
        results = self.registry.search_models(query)
        
        # Filter by access level
        if is_public:
            results = [m for m in results 
                      if self.registry.get_model(m).get("access_level") in 
                      [AccessLevel.PUBLIC.value, AccessLevel.SHARED.value]]
        
        # Sort by similarity score (number of matching tags)
        def similarity_score(model_id):
            model = self.registry.get_model(model_id)
            model_tags = set(model.get("tags", []))
            base_tags = set(base_info.get("tags", []))
            return len(model_tags & base_tags) / max(len(model_tags | base_tags), 1)
        
        results.sort(key=similarity_score, reverse=True)
        return results
    
    def recommend_models_for_task(self, task_description: str, 
                                 tenant_id: str = None) -> List[str]:
        """Recommend models for task (based on tags/metadata)"""
        # Extract keywords from description (simple keyword matching)
        keywords = set(task_description.lower().split())
        
        query = {"tags": list(keywords)}
        if tenant_id:
            query["tenant_id"] = tenant_id
        
        results = self.registry.search_models(query)
        return results[:10]  # Return top 10
    
    def discover_trending_models(self, num_top: int = 10,
                                time_window_days: int = 7) -> List[str]:
        """Get trending models"""
        # Simple: models registered recently
        recent_models = [
            (mid, model.get("registered_at"))
            for mid, model in self.registry.models.items()
        ]
        
        recent_models.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in recent_models[:num_top]]


@dataclass
class RegistryConfig:
    """Configuration for registry"""
    max_models_per_tenant: int = 1000
    max_versions_per_model: int = 20
    auto_cleanup_days: int = 90  # Auto-archive old failed models
    enable_metrics_caching: bool = True
    cache_ttl_minutes: int = 60

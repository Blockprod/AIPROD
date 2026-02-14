"""
User Model Management and Deduplication

Manages per-user custom models, handles model deduplication,
and optimizes storage through shared weights.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


class UserModelTier(Enum):
    """User tier affecting model limits"""
    FREE = (2, 10)  # (max_models, max_monthly_tokens)
    STARTER = (5, 100000)
    PRO = (20, 1000000)
    ENTERPRISE = (1000, None)


@dataclass
class UserModelQuota:
    """Quota for user's LoRA models"""
    user_id: str
    tier: UserModelTier
    models_created: int = 0
    models_deployed: int = 0
    storage_used_mb: float = 0.0
    training_hours_used: float = 0.0
    
    @property
    def max_models(self) -> int:
        """Maximum models allowed"""
        return self.tier.value[0]
    
    @property
    def max_monthly_tokens(self) -> Optional[int]:
        """Maximum tokens per month"""
        return self.tier.value[1]
    
    def can_create_model(self) -> bool:
        """Check if user can create more models"""
        return self.models_created < self.max_models
    
    def can_train(self, projected_storage_mb: float) -> bool:
        """Check if user has resources to train"""
        # Simple check (in real system: check quota limits)
        return self.can_create_model()


@dataclass
class UserModel:
    """User's custom LoRA model"""
    model_id: str
    user_id: str
    tenant_id: str
    name: str
    description: str
    base_model: str
    created_at: datetime
    last_trained_at: Optional[datetime] = None
    num_parameters: int = 0
    storage_mb: float = 0.0
    training_samples: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_shared: bool = False
    shared_with_users: Set[str] = field(default_factory=set)
    is_published: bool = False  # Public or organization-wide
    tags: Set[str] = field(default_factory=set)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get model storage information"""
        return {
            "model_id": self.model_id,
            "storage_mb": self.storage_mb,
            "num_parameters": self.num_parameters,
            "estimated_inference_memory_mb": self.storage_mb * 1.5  # Rough estimate
        }


class UserModelManager:
    """Manages models for individual users"""
    
    def __init__(self, user_id: str, tenant_id: str):
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.models: Dict[str, UserModel] = {}
        self.quota = UserModelQuota(user_id=user_id, tier=UserModelTier.FREE)
    
    def create_model(self, model_id: str, name: str, 
                    base_model: str, **metadata) -> Optional[UserModel]:
        """Create new user model"""
        if not self.quota.can_create_model():
            return None  # Quota exceeded
        
        model = UserModel(
            model_id=model_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            name=name,
            description=metadata.get("description", ""),
            base_model=base_model,
            created_at=datetime.utcnow(),
            **{k: v for k, v in metadata.items() if k not in ["description"]}
        )
        
        self.models[model_id] = model
        self.quota.models_created += 1
        
        return model
    
    def get_model(self, model_id: str) -> Optional[UserModel]:
        """Get user's model"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[UserModel]:
        """List all user models"""
        return list(self.models.values())
    
    def delete_model(self, model_id: str) -> bool:
        """Delete user model"""
        if model_id in self.models:
            model = self.models[model_id]
            del self.models[model_id]
            self.quota.storage_used_mb -= model.storage_mb
            return True
        return False
    
    def share_model(self, model_id: str, user_ids: List[str]) -> bool:
        """Share model with other users"""
        if model_id not in self.models:
            return False
        
        self.models[model_id].is_shared = True
        self.models[model_id].shared_with_users = set(user_ids)
        return True
    
    def update_quota(self, tier: UserModelTier):
        """Update user tier and quotas"""
        self.quota.tier = tier
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get user usage statistics"""
        return {
            "user_id": self.user_id,
            "tier": self.quota.tier.name,
            "models_created": self.quota.models_created,
            "max_models": self.quota.max_models,
            "models_deployed": self.quota.models_deployed,
            "storage_used_mb": self.quota.storage_used_mb,
            "num_models": len(self.models),
            "quota_utilization": min(1.0, self.quota.models_created / max(self.quota.max_models, 1))
        }


class ModelDeduplication:
    """Detects and deduplicates similar models"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.model_signatures: Dict[str, str] = {}  # model_id -> signature
    
    def compute_signature(self, model_weights: Dict[str, Any]) -> str:
        """Compute signature of model weights"""
        import hashlib
        
        # Simple signature: hash of parameter names and shapes
        sig_parts = []
        for name, weights in sorted(model_weights.items()):
            if hasattr(weights, '__len__'):
                try:
                    shape_str = f"{name}:{len(weights)}"
                except:
                    shape_str = f"{name}:scalar"
            else:
                shape_str = f"{name}:scalar"
            sig_parts.append(shape_str)
        
        signature_str = "|".join(sig_parts)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def find_similar_models(self, model_id: str,
                           all_models: Dict[str, UserModel]) -> List[Tuple[str, float]]:
        """Find models similar to given model"""
        if model_id not in self.model_signatures:
            return []
        
        target_sig = self.model_signatures[model_id]
        similar = []
        
        for other_id, other_sig in self.model_signatures.items():
            if other_id == model_id:
                continue
            
            # Compute similarity (simple: exact match or prefix match)
            if other_sig == target_sig:
                similarity = 1.0
            elif (target_sig[:16] == other_sig[:16] and 
                  len(target_sig) == len(other_sig)):
                similarity = 0.95
            else:
                continue
            
            if similarity >= self.similarity_threshold:
                similar.append((other_id, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def recommend_deduplication(self, all_models: Dict[str, UserModel]) -> List[List[str]]:
        """Recommend groups of models to deduplicate"""
        duplicates_groups = []
        seen_signatures = {}
        
        for model_id, model in all_models.items():
            # Create dummy weights dict for signature
            weights = {"base": None}
            sig = self.compute_signature(weights)
            
            if sig not in seen_signatures:
                seen_signatures[sig] = []
            seen_signatures[sig].append(model_id)
        
        # Return groups with more than one model
        for sig, model_ids in seen_signatures.items():
            if len(model_ids) > 1:
                duplicates_groups.append(model_ids)
        
        return duplicates_groups
    
    def estimate_deduplication_savings(self, size_per_model_mb: float,
                                      num_duplicates: int) -> float:
        """Estimate storage savings from deduplication"""
        # Keep one model, share weights with others
        savings = size_per_model_mb * (num_duplicates - 1)
        return max(0, savings)


class TenantModelManager:
    """Manages all user models for a tenant"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.user_managers: Dict[str, UserModelManager] = {}
        self.all_models: Dict[str, UserModel] = {}
        self.deduplication = ModelDeduplication()
    
    def get_user_manager(self, user_id: str) -> UserModelManager:
        """Get or create manager for user"""
        if user_id not in self.user_managers:
            self.user_managers[user_id] = UserModelManager(user_id, self.tenant_id)
        return self.user_managers[user_id]
    
    def create_user_model(self, user_id: str, model_id: str, 
                         name: str, base_model: str) -> Optional[UserModel]:
        """Create model for user"""
        manager = self.get_user_manager(user_id)
        model = manager.create_model(model_id, name, base_model)
        
        if model:
            self.all_models[model_id] = model
        
        return model
    
    def list_user_models(self, user_id: str) -> List[UserModel]:
        """List models for user"""
        manager = self.get_user_manager(user_id)
        return manager.list_models()
    
    def get_deduplication_report(self) -> Dict[str, Any]:
        """Get deduplication opportunities"""
        duplicate_groups = self.deduplication.recommend_deduplication(self.all_models)
        
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
        potential_savings = sum(
            self.deduplication.estimate_deduplication_savings(100.0, len(group))
            for group in duplicate_groups
        )
        
        return {
            "num_duplicate_groups": len(duplicate_groups),
            "total_duplicate_models": total_duplicates,
            "potential_storage_savings_mb": potential_savings,
            "duplicate_groups": duplicate_groups[:5]  # Top 5 groups
        }
    
    def get_tenant_stats(self) -> Dict[str, Any]:
        """Get tenant-wide statistics"""
        total_models = len(self.all_models)
        total_storage = sum(m.storage_mb for m in self.all_models.values())
        
        return {
            "tenant_id": self.tenant_id,
            "num_users": len(self.user_managers),
            "total_models": total_models,
            "total_storage_mb": total_storage,
            "avg_storage_per_model": total_storage / max(1, total_models)
        }

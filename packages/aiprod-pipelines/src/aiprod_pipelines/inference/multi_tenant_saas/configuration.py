"""
Configuration Management for Multi-Tenant SaaS.

Handles per-tenant settings, feature flags, and configuration
management with gradual rollout support.

Core Classes:
  - TenantConfig: Tenant-specific settings
  - FeatureFlag: Feature flag definition
  - FeatureFlagManager: Feature flag control
  - ConfigurationManager: Configuration lifecycle
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime
from enum import Enum
import threading
import json


class FeatureFlagType(str, Enum):
    """Feature flag types."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    RULE_BASED = "rule_based"


class RolloutStatus(str, Enum):
    """Feature rollout status."""
    PLANNING = "planning"
    ALPHA = "alpha"
    BETA = "beta"
    GRADUAL_ROLLOUT = "gradual_rollout"
    GENERAL_AVAILABILITY = "general_availability"
    DISCONTINUING = "discontinuing"
    DISCONTINUED = "discontinued"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    flag_id: str
    name: str
    description: str = ""
    
    enabled: bool = False
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    rollout_status: RolloutStatus = RolloutStatus.PLANNING
    
    # Percentage rollout (0-100)
    rollout_percentage: float = 0.0
    
    # User list for targeted rollout
    enabled_users: Set[str] = field(default_factory=set)
    enabled_tenants: Set[str] = field(default_factory=set)
    
    # Rule-based config
    rules: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    owner: str = ""
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_enabled_for_tenant(self, tenant_id: str) -> bool:
        """Check if flag is enabled for tenant."""
        if not self.enabled:
            return False
        
        if self.flag_type == FeatureFlagType.BOOLEAN:
            return True
        
        if self.flag_type == FeatureFlagType.USER_LIST:
            return tenant_id in self.enabled_tenants
        
        if self.flag_type == FeatureFlagType.PERCENTAGE:
            # Simple hash-based percentage rollout
            hash_val = hash(tenant_id) % 100
            return hash_val < self.rollout_percentage
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_id": self.flag_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "flag_type": self.flag_type.value,
            "rollout_status": self.rollout_status.value,
            "rollout_percentage": self.rollout_percentage,
            "tags": list(self.tags),
        }


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""
    tenant_id: str
    
    # Feature settings
    features: Dict[str, bool] = field(default_factory=dict)
    
    # Model settings
    default_model: str = "ltx-video-2"
    allowed_models: Set[str] = field(default_factory=lambda: {"ltx-video-2"})
    
    # Generation settings
    default_num_inference_steps: int = 30
    max_num_inference_steps: int = 100
    default_guidance_scale: float = 7.5
    
    # Storage settings
    storage_backend: str = "local"  # local, s3, gcs, azure
    max_storage_gb: float = 100.0
    retention_days: int = 90
    
    # API settings
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Version control
    config_version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Advanced settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def get_feature_enabled(self, feature: str, default: bool = False) -> bool:
        """Get feature enabled status."""
        return self.features.get(feature, default)
    
    def set_feature_enabled(self, feature: str, enabled: bool) -> None:
        """Set feature enabled status."""
        self.features[feature] = enabled
        self.updated_at = datetime.utcnow()
    
    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Get custom setting."""
        return self.custom_settings.get(key, default)
    
    def set_custom_setting(self, key: str, value: Any) -> None:
        """Set custom setting."""
        self.custom_settings[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "features": self.features,
            "default_model": self.default_model,
            "allowed_models": list(self.allowed_models),
            "default_num_inference_steps": self.default_num_inference_steps,
            "default_guidance_scale": self.default_guidance_scale,
            "storage_backend": self.storage_backend,
            "max_storage_gb": self.max_storage_gb,
            "retention_days": self.retention_days,
            "updated_at": self.updated_at.isoformat(),
        }


class FeatureFlagManager:
    """Manages feature flags and gradual rollout."""
    
    def __init__(self):
        """Initialize feature flag manager."""
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()
    
    def create_flag(self, flag: FeatureFlag) -> None:
        """Create a new feature flag."""
        with self._lock:
            self._flags[flag.flag_id] = flag
    
    def update_flag(self, flag: FeatureFlag) -> None:
        """Update a feature flag."""
        with self._lock:
            self._flags[flag.flag_id] = flag
            flag.updated_at = datetime.utcnow()
    
    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Get feature flag."""
        with self._lock:
            return self._flags.get(flag_id)
    
    def enable_flag(self, flag_id: str) -> bool:
        """Enable a feature flag."""
        with self._lock:
            if flag_id in self._flags:
                self._flags[flag_id].enabled = True
                self._flags[flag_id].updated_at = datetime.utcnow()
                return True
        return False
    
    def disable_flag(self, flag_id: str) -> bool:
        """Disable a feature flag."""
        with self._lock:
            if flag_id in self._flags:
                self._flags[flag_id].enabled = False
                self._flags[flag_id].updated_at = datetime.utcnow()
                return True
        return False
    
    def set_rollout_percentage(self, flag_id: str, percentage: float) -> bool:
        """Set percentage rollout for flag."""
        with self._lock:
            if flag_id in self._flags:
                flag = self._flags[flag_id]
                flag.rollout_percentage = max(0.0, min(100.0, percentage))
                flag.updated_at = datetime.utcnow()
                return True
        return False
    
    def add_enabled_tenant(self, flag_id: str, tenant_id: str) -> bool:
        """Add tenant to enabled tenants."""
        with self._lock:
            if flag_id in self._flags:
                self._flags[flag_id].enabled_tenants.add(tenant_id)
                self._flags[flag_id].updated_at = datetime.utcnow()
                return True
        return False
    
    def remove_enabled_tenant(self, flag_id: str, tenant_id: str) -> bool:
        """Remove tenant from enabled tenants."""
        with self._lock:
            if flag_id in self._flags:
                self._flags[flag_id].enabled_tenants.discard(tenant_id)
                self._flags[flag_id].updated_at = datetime.utcnow()
                return True
        return False
    
    def is_feature_enabled(self, flag_id: str, tenant_id: str) -> bool:
        """Check if feature is enabled for tenant."""
        with self._lock:
            if flag_id not in self._flags:
                return False
            return self._flags[flag_id].is_enabled_for_tenant(tenant_id)
    
    def get_all_flags(self) -> List[FeatureFlag]:
        """Get all feature flags."""
        with self._lock:
            return list(self._flags.values())
    
    def get_flags_for_tenant(self, tenant_id: str) -> List[FeatureFlag]:
        """Get enabled flags for tenant."""
        with self._lock:
            flags = []
            for flag in self._flags.values():
                if flag.is_enabled_for_tenant(tenant_id):
                    flags.append(flag)
            return flags


class ConfigurationManager:
    """Manages tenant configurations."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._configs: Dict[str, TenantConfig] = {}
        self._default_config = None
        self._lock = threading.RLock()
    
    def set_default_config(self, config: TenantConfig) -> None:
        """Set default configuration template."""
        with self._lock:
            self._default_config = config
    
    def create_config(self, tenant_id: str) -> TenantConfig:
        """Create configuration for tenant."""
        with self._lock:
            if tenant_id in self._configs:
                return self._configs[tenant_id]
            
            # Create from default or create new
            if self._default_config:
                config = TenantConfig(tenant_id=tenant_id)
                config.features = dict(self._default_config.features)
                config.allowed_models = set(self._default_config.allowed_models)
            else:
                config = TenantConfig(tenant_id=tenant_id)
            
            self._configs[tenant_id] = config
            return config
    
    def get_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        with self._lock:
            if tenant_id not in self._configs:
                return self.create_config(tenant_id)
            return self._configs[tenant_id]
    
    def update_config(self, config: TenantConfig) -> None:
        """Update tenant configuration."""
        with self._lock:
            self._configs[config.tenant_id] = config
            config.updated_at = datetime.utcnow()
            config.config_version += 1
    
    def reset_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Reset configuration to default."""
        with self._lock:
            if self._default_config:
                config = TenantConfig(tenant_id=tenant_id)
                config.features = dict(self._default_config.features)
                config.allowed_models = set(self._default_config.allowed_models)
                self._configs[tenant_id] = config
                return config
        return None
    
    def get_all_configs(self) -> List[TenantConfig]:
        """Get all tenant configurations."""
        with self._lock:
            return list(self._configs.values())

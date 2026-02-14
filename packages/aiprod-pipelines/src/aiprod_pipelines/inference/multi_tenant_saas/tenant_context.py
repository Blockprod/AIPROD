"""
Tenant Context Management for Multi-Tenant SaaS Platform.

Provides core tenant identification, isolation, and metadata management.
Each request is associated with a tenant context that controls resource access,
feature availability, and configuration.

Core Classes:
  - TenantMetadata: Tenant profile and configuration
  - TenantContext: Runtime context for a single request
  - TenantContextManager: Lifecycle management
  - TenantRegistry: Central tenant tracking
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List, Any
from enum import Enum
import time
import threading
from datetime import datetime, timedelta


class TenantTier(str, Enum):
    """Subscription tier levels."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    ARCHIVED = "archived"


@dataclass
class TenantQuotas:
    """Resource quotas for a tenant."""
    max_concurrent_jobs: int = 5
    max_videos_per_month: int = 100
    max_video_duration_seconds: int = 600
    max_resolution_pixels: int = 1920 * 1080
    storage_gb: float = 50.0
    api_calls_per_minute: int = 100
    batch_job_size: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_videos_per_month": self.max_videos_per_month,
            "max_video_duration_seconds": self.max_video_duration_seconds,
            "max_resolution_pixels": self.max_resolution_pixels,
            "storage_gb": self.storage_gb,
            "api_calls_per_minute": self.api_calls_per_minute,
            "batch_job_size": self.batch_job_size,
        }


@dataclass
class TenantMetadata:
    """Complete tenant profile and configuration."""
    tenant_id: str
    organization_name: str
    tier: TenantTier = TenantTier.FREE
    status: TenantStatus = TenantStatus.ACTIVE
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    features: Set[str] = field(default_factory=set)  # e.g., {"lora", "quantization", "distillation"}
    created_at: datetime = field(default_factory=datetime.utcnow)
    subscription_end_date: Optional[datetime] = None
    billing_contact_email: str = ""
    api_key_prefix: str = ""  # First 6 chars of hashed API key
    custom_domain: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE
    
    def is_trial(self) -> bool:
        """Check if tenant is in trial period."""
        return self.status == TenantStatus.TRIAL
    
    def trial_remaining_days(self) -> Optional[int]:
        """Days remaining in trial period."""
        if not self.is_trial() or not self.subscription_end_date:
            return None
        remaining = (self.subscription_end_date - datetime.utcnow()).days
        return max(0, remaining)
    
    def has_feature(self, feature: str) -> bool:
        """Check if tenant has access to feature."""
        return feature in self.features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "organization_name": self.organization_name,
            "tier": self.tier.value,
            "status": self.status.value,
            "quotas": self.quotas.to_dict(),
            "features": list(self.features),
            "created_at": self.created_at.isoformat(),
            "subscription_end_date": self.subscription_end_date.isoformat() if self.subscription_end_date else None,
            "billing_contact_email": self.billing_contact_email,
            "custom_domain": self.custom_domain,
            "trial_remaining_days": self.trial_remaining_days(),
        }


@dataclass
class TenantContext:
    """Runtime context for a single request/session."""
    tenant_id: str
    tenant_metadata: TenantMetadata
    request_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    api_key_used: bool = False
    session_token: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_tenant_id(self) -> str:
        """Get current tenant ID."""
        return self.tenant_id
    
    def get_tier(self) -> TenantTier:
        """Get tenant tier."""
        return self.tenant_metadata.tier
    
    def get_quotas(self) -> TenantQuotas:
        """Get tenant quotas."""
        return self.tenant_metadata.quotas
    
    def set_custom_data(self, key: str, value: Any) -> None:
        """Set custom request data."""
        self.custom_data[key] = value
    
    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """Get custom request data."""
        return self.custom_data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tier": self.tenant_metadata.tier.value,
            "online_seconds": (datetime.utcnow() - self.timestamp).total_seconds(),
        }


class TenantContextManager:
    """Manages tenant context lifecycle."""
    
    def __init__(self):
        """Initialize context manager."""
        self._context_stack: Dict[int, List[TenantContext]] = {}
        self._lock = threading.RLock()
    
    def push_context(self, context: TenantContext) -> None:
        """Push context onto thread stack."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._context_stack:
                self._context_stack[thread_id] = []
            self._context_stack[thread_id].append(context)
    
    def pop_context(self) -> Optional[TenantContext]:
        """Pop context from thread stack."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._context_stack and self._context_stack[thread_id]:
                return self._context_stack[thread_id].pop()
        return None
    
    def get_current_context(self) -> Optional[TenantContext]:
        """Get current context without removing."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._context_stack and self._context_stack[thread_id]:
                return self._context_stack[thread_id][-1]
        return None
    
    def context_depth(self) -> int:
        """Get current context stack depth."""
        thread_id = threading.get_ident()
        with self._lock:
            return len(self._context_stack.get(thread_id, []))


class TenantRegistry:
    """Central registry for all active tenants."""
    
    def __init__(self):
        """Initialize registry."""
        self._tenants: Dict[str, TenantMetadata] = {}
        self._tenant_by_api_key: Dict[str, str] = {}  # api_key -> tenant_id
        self._tenant_by_domain: Dict[str, str] = {}  # domain -> tenant_id
        self._lock = threading.RLock()
        self._last_updated: Dict[str, datetime] = {}
    
    def register_tenant(self, metadata: TenantMetadata) -> None:
        """Register a new tenant."""
        with self._lock:
            self._tenants[metadata.tenant_id] = metadata
            if metadata.api_key_prefix:
                self._tenant_by_api_key[metadata.api_key_prefix] = metadata.tenant_id
            if metadata.custom_domain:
                self._tenant_by_domain[metadata.custom_domain] = metadata.tenant_id
            self._last_updated[metadata.tenant_id] = datetime.utcnow()
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantMetadata]:
        """Get tenant metadata by ID."""
        with self._lock:
            return self._tenants.get(tenant_id)
    
    def get_tenant_by_api_key_prefix(self, key_prefix: str) -> Optional[TenantMetadata]:
        """Get tenant by API key prefix."""
        with self._lock:
            tenant_id = self._tenant_by_api_key.get(key_prefix)
            return self._tenants.get(tenant_id) if tenant_id else None
    
    def get_tenant_by_domain(self, domain: str) -> Optional[TenantMetadata]:
        """Get tenant by custom domain."""
        with self._lock:
            tenant_id = self._tenant_by_domain.get(domain)
            return self._tenants.get(tenant_id) if tenant_id else None
    
    def update_tenant_status(self, tenant_id: str, status: TenantStatus) -> bool:
        """Update tenant status."""
        with self._lock:
            if tenant_id in self._tenants:
                self._tenants[tenant_id].status = status
                self._last_updated[tenant_id] = datetime.utcnow()
                return True
        return False
    
    def update_tenant_tier(self, tenant_id: str, tier: TenantTier, quotas: Optional[TenantQuotas] = None) -> bool:
        """Update tenant tier and optionally quotas."""
        with self._lock:
            if tenant_id in self._tenants:
                self._tenants[tenant_id].tier = tier
                if quotas:
                    self._tenants[tenant_id].quotas = quotas
                self._last_updated[tenant_id] = datetime.utcnow()
                return True
        return False
    
    def add_feature(self, tenant_id: str, feature: str) -> bool:
        """Add feature to tenant."""
        with self._lock:
            if tenant_id in self._tenants:
                self._tenants[tenant_id].features.add(feature)
                self._last_updated[tenant_id] = datetime.utcnow()
                return True
        return False
    
    def remove_feature(self, tenant_id: str, feature: str) -> bool:
        """Remove feature from tenant."""
        with self._lock:
            if tenant_id in self._tenants:
                self._tenants[tenant_id].features.discard(feature)
                self._last_updated[tenant_id] = datetime.utcnow()
                return True
        return False
    
    def list_active_tenants(self) -> List[TenantMetadata]:
        """List all active tenants."""
        with self._lock:
            return [t for t in self._tenants.values() if t.is_active()]
    
    def get_tenant_count(self) -> int:
        """Get total tenant count."""
        with self._lock:
            return len(self._tenants)
    
    def get_age_seconds(self, tenant_id: str) -> Optional[float]:
        """Get tenant age in seconds since registration."""
        with self._lock:
            if tenant_id in self._last_updated:
                return (datetime.utcnow() - self._last_updated[tenant_id]).total_seconds()
        return None


# Global registry instance
_global_registry = TenantRegistry()
_global_context_manager = TenantContextManager()


def get_tenant_registry() -> TenantRegistry:
    """Get global tenant registry."""
    return _global_registry


def get_context_manager() -> TenantContextManager:
    """Get global context manager."""
    return _global_context_manager


def get_current_tenant_context() -> Optional[TenantContext]:
    """Get current tenant context from thread-local storage."""
    return get_context_manager().get_current_context()


class TenantContextualResource:
    """Base class for tenant-aware resources."""
    
    def __init__(self, tenant_id: str):
        """Initialize with tenant ID."""
        self.tenant_id = tenant_id
        self._registry = get_tenant_registry()
        self._metadata = self._registry.get_tenant(tenant_id)
    
    def get_metadata(self) -> Optional[TenantMetadata]:
        """Get tenant metadata."""
        return self._metadata
    
    def is_tenant_active(self) -> bool:
        """Check if tenant is active."""
        return self._metadata.is_active() if self._metadata else False
    
    def check_feature_access(self, feature: str) -> bool:
        """Check if tenant has feature access."""
        return self._metadata.has_feature(feature) if self._metadata else False

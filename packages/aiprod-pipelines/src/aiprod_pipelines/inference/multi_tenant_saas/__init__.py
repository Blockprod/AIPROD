"""
Multi-Tenant SaaS Platform Module.

Complete SaaS infrastructure with tenant isolation, authentication,
RBAC, billing, job management, and monitoring.

Main Components:
  - Tenant Management: Isolation and metadata
  - Authentication: API keys, JWT, sessions
  - Access Control: RBAC with roles and permissions
  - Usage Tracking: Metering and billing integration
  - Billing: Invoicing and pricing
  - API Gateway: Rate limiting and request validation
  - Job Management: Long-running task scheduling
  - Configuration: Feature flags and settings
  - Monitoring: Health and analytics
"""

# Tenant Context
from .tenant_context import (
    TenantTier,
    TenantStatus,
    TenantQuotas,
    TenantMetadata,
    TenantContext,
    TenantContextManager,
    TenantRegistry,
    get_tenant_registry,
    get_context_manager,
    get_current_tenant_context,
    TenantContextualResource,
)

# Authentication
from .authentication import (
    TokenType,
    AuthenticationMethod,
    APIKey,
    JWTToken,
    JWTTokenManager,
    SessionManager,
    AuthenticationManager,
)

# Access Control
from .access_control import (
    ResourceType,
    Action,
    RoleType,
    Permission,
    Role,
    UserRole,
    RoleBasedAccessControl,
    PermissionChecker,
)

# Usage Tracking
from .usage_tracking import (
    UsageEventType,
    UsageEvent,
    UsageMetrics,
    UsageEventLogger,
    MeteringEngine,
    UsageAggregator,
)

# Billing
from .billing import (
    BillingCycle,
    InvoiceStatus,
    PricingModel,
    SubscriptionPlan,
    LineItem,
    Invoice,
    BillingCalculator,
    BillingPortal,
)

# API Gateway
from .api_gateway import (
    RateLimitWindow,
    RateLimitConfig,
    RequestQuota,
    APIRequest,
    APIResponse,
    RateLimiter,
    RequestValidator,
    APIGateway,
    GatewayMetrics,
)

# Job Manager
from .job_manager import (
    JobStatus,
    JobPriority,
    BatchJob,
    JobResult,
    JobProgressTracker,
    JobScheduler,
    JobManagementPortal,
)

# Configuration
from .configuration import (
    FeatureFlagType,
    RolloutStatus,
    FeatureFlag,
    TenantConfig,
    FeatureFlagManager,
    ConfigurationManager,
)

# Monitoring
from .monitoring import (
    HealthStatus,
    MetricSnapshot,
    TenantMetrics,
    MetricsCollector,
    AnomalyDetector,
    AnalyticsCollector,
    HealthMonitor,
)

__all__ = [
    # Tenant Context
    "TenantTier",
    "TenantStatus",
    "TenantQuotas",
    "TenantMetadata",
    "TenantContext",
    "TenantContextManager",
    "TenantRegistry",
    "get_tenant_registry",
    "get_context_manager",
    "get_current_tenant_context",
    "TenantContextualResource",
    # Authentication
    "TokenType",
    "AuthenticationMethod",
    "APIKey",
    "JWTToken",
    "JWTTokenManager",
    "SessionManager",
    "AuthenticationManager",
    # Access Control
    "ResourceType",
    "Action",
    "RoleType",
    "Permission",
    "Role",
    "UserRole",
    "RoleBasedAccessControl",
    "PermissionChecker",
    # Usage Tracking
    "UsageEventType",
    "UsageEvent",
    "UsageMetrics",
    "UsageEventLogger",
    "MeteringEngine",
    "UsageAggregator",
    # Billing
    "BillingCycle",
    "InvoiceStatus",
    "PricingModel",
    "SubscriptionPlan",
    "LineItem",
    "Invoice",
    "BillingCalculator",
    "BillingPortal",
    # API Gateway
    "RateLimitWindow",
    "RateLimitConfig",
    "RequestQuota",
    "APIRequest",
    "APIResponse",
    "RateLimiter",
    "RequestValidator",
    "APIGateway",
    "GatewayMetrics",
    # Job Manager
    "JobStatus",
    "JobPriority",
    "BatchJob",
    "JobResult",
    "JobProgressTracker",
    "JobScheduler",
    "JobManagementPortal",
    # Configuration
    "FeatureFlagType",
    "RolloutStatus",
    "FeatureFlag",
    "TenantConfig",
    "FeatureFlagManager",
    "ConfigurationManager",
    # Monitoring
    "HealthStatus",
    "MetricSnapshot",
    "TenantMetrics",
    "MetricsCollector",
    "AnomalyDetector",
    "AnalyticsCollector",
    "HealthMonitor",
]

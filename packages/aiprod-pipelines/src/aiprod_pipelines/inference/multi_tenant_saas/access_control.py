"""
Role-Based Access Control (RBAC) for Multi-Tenant SaaS.

Provides fine-grained permission management, role definitions,
and resource access verification.

Core Classes:
  - Permission: Individual permission definition
  - Role: Collection of permissions
  - RoleBasedAccessControl: RBAC engine
  - PermissionChecker: Request-level validation
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Tuple, Any
from enum import Enum
import threading


class ResourceType(str, Enum):
    """Resource types in the system."""
    VIDEO_PROJECT = "video_project"
    MODEL = "model"
    DATASET = "dataset"
    API_KEY = "api_key"
    BILLING = "billing"
    SETTINGS = "settings"
    USER_MANAGEMENT = "user_management"
    WEBHOOK = "webhook"
    INTEGRATION = "integration"


class Action(str, Enum):
    """Actions that can be performed on resources."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"
    EXPORT = "export"
    ADMIN = "admin"


class RoleType(str, Enum):
    """Predefined role types."""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"
    CUSTOM = "custom"


@dataclass
class Permission:
    """Individual permission."""
    resource: ResourceType
    action: Action
    
    def __hash__(self) -> int:
        """Hash for set operations."""
        return hash((self.resource.value, self.action.value))
    
    def __eq__(self, other: object) -> bool:
        """Equality check."""
        if not isinstance(other, Permission):
            return False
        return self.resource == other.resource and self.action == other.action
    
    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.resource.value}:{self.action.value}"
    
    @staticmethod
    def from_string(permission_str: str) -> Optional["Permission"]:
        """Create from string representation."""
        try:
            resource_str, action_str = permission_str.split(":")
            resource = ResourceType(resource_str)
            action = Action(action_str)
            return Permission(resource, action)
        except (ValueError, KeyError):
            return None


@dataclass
class Role:
    """Role definition with permissions."""
    role_id: str
    name: str
    role_type: RoleType
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    is_system_role: bool = False  # Cannot be modified
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to role."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from role."""
        self.permissions.discard(permission)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission."""
        return permission in self.permissions
    
    def has_resource_action(self, resource: ResourceType, action: Action) -> bool:
        """Check if role can perform action on resource."""
        return Permission(resource, action) in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role_id": self.role_id,
            "name": self.name,
            "role_type": self.role_type.value,
            "permissions": [p.to_string() for p in self.permissions],
            "description": self.description,
            "is_system_role": self.is_system_role,
        }


@dataclass
class UserRole:
    """User-role assignment in a tenant."""
    user_id: str
    tenant_id: str
    roles: Set[str] = field(default_factory=set)  # role_ids
    custom_permissions: Set[Permission] = field(default_factory=set)
    resource_permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    created_at: str = ""
    
    def add_role(self, role_id: str) -> None:
        """Add role to user."""
        self.roles.add(role_id)
    
    def remove_role(self, role_id: str) -> None:
        """Remove role from user."""
        self.roles.discard(role_id)
    
    def add_custom_permission(self, permission: Permission) -> None:
        """Add custom permission directly."""
        self.custom_permissions.add(permission)
    
    def add_resource_permission(self, resource_id: str, permission: Permission) -> None:
        """Add resource-specific permission."""
        if resource_id not in self.resource_permissions:
            self.resource_permissions[resource_id] = set()
        self.resource_permissions[resource_id].add(permission)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "roles": list(self.roles),
            "custom_permissions": [p.to_string() for p in self.custom_permissions],
            "resource_count": len(self.resource_permissions),
        }


def _create_admin_role() -> Role:
    """Create system admin role."""
    admin_role = Role(
        role_id="system_admin",
        name="Administrator",
        role_type=RoleType.ADMIN,
        is_system_role=True,
        description="Full system access",
    )
    # Admin can do everything
    for resource in ResourceType:
        for action in Action:
            admin_role.add_permission(Permission(resource, action))
    return admin_role


def _create_manager_role() -> Role:
    """Create system manager role."""
    manager_role = Role(
        role_id="system_manager",
        name="Manager",
        role_type=RoleType.MANAGER,
        is_system_role=True,
        description="Team and project management",
    )
    # Manager can manage most resources except billing/admin
    for resource in ResourceType:
        if resource not in [ResourceType.BILLING, ResourceType.USER_MANAGEMENT]:
            for action in [Action.CREATE, Action.READ, Action.UPDATE, Action.EXECUTE]:
                manager_role.add_permission(Permission(resource, action))
    return manager_role


def _create_user_role() -> Role:
    """Create system user role."""
    user_role = Role(
        role_id="system_user",
        name="User",
        role_type=RoleType.USER,
        is_system_role=True,
        description="Standard user access",
    )
    # User can create, read, update their own resources
    for resource in [
        ResourceType.VIDEO_PROJECT,
        ResourceType.MODEL,
        ResourceType.DATASET,
    ]:
        for action in [Action.CREATE, Action.READ, Action.UPDATE, Action.EXECUTE]:
            user_role.add_permission(Permission(resource, action))
    return user_role


def _create_viewer_role() -> Role:
    """Create system viewer role."""
    viewer_role = Role(
        role_id="system_viewer",
        name="Viewer",
        role_type=RoleType.VIEWER,
        is_system_role=True,
        description="Read-only access",
    )
    # Viewer can only read
    for resource in ResourceType:
        viewer_role.add_permission(Permission(resource, Action.READ))
    return viewer_role


class RoleBasedAccessControl:
    """RBAC engine for permission management."""
    
    def __init__(self):
        """Initialize RBAC."""
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[Tuple[str, str], UserRole] = {}  # (tenant_id, user_id) -> UserRole
        self._lock = threading.RLock()
        
        # Register system roles
        self._roles["system_admin"] = _create_admin_role()
        self._roles["system_manager"] = _create_manager_role()
        self._roles["system_user"] = _create_user_role()
        self._roles["system_viewer"] = _create_viewer_role()
    
    def create_custom_role(
        self,
        tenant_id: str,
        name: str,
        permissions: Optional[Set[Permission]] = None,
        description: str = "",
    ) -> Role:
        """Create a custom role for a tenant."""
        role_id = f"{tenant_id}_{name.lower().replace(' ', '_')}"
        
        role = Role(
            role_id=role_id,
            name=name,
            role_type=RoleType.CUSTOM,
            permissions=permissions or set(),
            description=description,
            is_system_role=False,
        )
        
        with self._lock:
            self._roles[role_id] = role
        
        return role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        with self._lock:
            return self._roles.get(role_id)
    
    def assign_role_to_user(self, tenant_id: str, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        with self._lock:
            if role_id not in self._roles:
                return False
            
            key = (tenant_id, user_id)
            if key not in self._user_roles:
                self._user_roles[key] = UserRole(user_id, tenant_id)
            
            self._user_roles[key].add_role(role_id)
            return True
    
    def revoke_role_from_user(self, tenant_id: str, user_id: str, role_id: str) -> bool:
        """Revoke role from user."""
        with self._lock:
            key = (tenant_id, user_id)
            if key in self._user_roles:
                self._user_roles[key].remove_role(role_id)
                return True
        return False
    
    def get_user_roles(self, tenant_id: str, user_id: str) -> List[Role]:
        """Get all roles for a user."""
        with self._lock:
            key = (tenant_id, user_id)
            if key not in self._user_roles:
                return []
            
            user_role = self._user_roles[key]
            return [
                self._roles[role_id]
                for role_id in user_role.roles
                if role_id in self._roles
            ]
    
    def check_permission(
        self,
        tenant_id: str,
        user_id: str,
        resource: ResourceType,
        action: Action,
    ) -> bool:
        """Check if user has permission."""
        permission = Permission(resource, action)
        
        with self._lock:
            key = (tenant_id, user_id)
            
            # Check direct custom permissions
            if key in self._user_roles:
                user_role = self._user_roles[key]
                if permission in user_role.custom_permissions:
                    return True
            
            # Check role permissions
            roles = self.get_user_roles(tenant_id, user_id)
            for role in roles:
                if role.has_permission(permission):
                    return True
        
        return False
    
    def check_resource_permission(
        self,
        tenant_id: str,
        user_id: str,
        resource_id: str,
        resource_type: ResourceType,
        action: Action,
    ) -> bool:
        """Check resource-specific permission."""
        permission = Permission(resource_type, action)
        
        with self._lock:
            key = (tenant_id, user_id)
            if key in self._user_roles:
                user_role = self._user_roles[key]
                if resource_id in user_role.resource_permissions:
                    if permission in user_role.resource_permissions[resource_id]:
                        return True
            
            # Fall back to general permission check
            return self.check_permission(tenant_id, user_id, resource_type, action)
    
    def grant_resource_access(
        self,
        tenant_id: str,
        user_id: str,
        resource_id: str,
        permissions: Set[Permission],
    ) -> bool:
        """Grant resource-specific access."""
        with self._lock:
            key = (tenant_id, user_id)
            if key not in self._user_roles:
                self._user_roles[key] = UserRole(user_id, tenant_id)
            
            user_role = self._user_roles[key]
            for permission in permissions:
                user_role.add_resource_permission(resource_id, permission)
            
            return True


class PermissionChecker:
    """Utility class for per-request permission validation."""
    
    def __init__(self, rbac: RoleBasedAccessControl):
        """Initialize permission checker."""
        self.rbac = rbac
    
    def require_permission(
        self,
        tenant_id: str,
        user_id: str,
        resource: ResourceType,
        action: Action,
    ) -> Tuple[bool, str]:
        """Check permission and return (allowed, message)."""
        if self.rbac.check_permission(tenant_id, user_id, resource, action):
            return True, "Permission granted"
        
        return False, f"Permission denied for {resource.value}:{action.value}"
    
    def require_any_role(
        self,
        tenant_id: str,
        user_id: str,
        required_roles: List[str],
    ) -> Tuple[bool, str]:
        """Check if user has any of required roles."""
        user_roles = {role.role_id for role in self.rbac.get_user_roles(tenant_id, user_id)}
        
        if any(role_id in user_roles for role_id in required_roles):
            return True, "Required role found"
        
        return False, f"User does not have any required role"
    
    def require_all_roles(
        self,
        tenant_id: str,
        user_id: str,
        required_roles: List[str],
    ) -> Tuple[bool, str]:
        """Check if user has all required roles."""
        user_roles = {role.role_id for role in self.rbac.get_user_roles(tenant_id, user_id)}
        
        if all(role_id in user_roles for role_id in required_roles):
            return True, "All required roles found"
        
        missing = set(required_roles) - user_roles
        return False, f"User missing roles: {missing}"

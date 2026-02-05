"""
Role-Based Access Control (RBAC) infrastructure for AIPROD
Provides role definitions, permission management, and access control
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import json
from datetime import datetime


class Role(Enum):
    """System roles"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"


class Permission(Enum):
    """Fine-grained permissions"""
    # Read operations
    READ_JOBS = "read:jobs"
    READ_RESULTS = "read:results"
    READ_LOGS = "read:logs"
    READ_METRICS = "read:metrics"
    
    # Write operations
    CREATE_JOB = "create:job"
    UPDATE_JOB = "update:job"
    DELETE_JOB = "delete:job"
    
    # Admin operations
    MANAGE_USERS = "manage:users"
    MANAGE_ROLES = "manage:roles"
    MANAGE_SETTINGS = "manage:settings"
    VIEW_AUDIT_LOG = "view:audit_log"
    
    # Service operations
    SERVICE_CALL = "service:call"
    SERVICE_WEBHOOK = "service:webhook"


class RBACConfig:
    """Central RBAC configuration and permission matrix"""
    
    # Define which permissions each role has
    ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.ADMIN: {
            # Admins can do everything
            Permission.READ_JOBS,
            Permission.READ_RESULTS,
            Permission.READ_LOGS,
            Permission.READ_METRICS,
            Permission.CREATE_JOB,
            Permission.UPDATE_JOB,
            Permission.DELETE_JOB,
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.MANAGE_SETTINGS,
            Permission.VIEW_AUDIT_LOG,
            Permission.SERVICE_CALL,
            Permission.SERVICE_WEBHOOK,
        },
        Role.USER: {
            # Users can perform operations on their own jobs
            Permission.READ_JOBS,
            Permission.READ_RESULTS,
            Permission.READ_LOGS,
            Permission.CREATE_JOB,
            Permission.UPDATE_JOB,
            Permission.DELETE_JOB,
        },
        Role.VIEWER: {
            # Viewers can only read
            Permission.READ_JOBS,
            Permission.READ_RESULTS,
            Permission.READ_LOGS,
            Permission.READ_METRICS,
        },
        Role.SERVICE: {
            # Services can call webhooks and service endpoints
            Permission.SERVICE_CALL,
            Permission.SERVICE_WEBHOOK,
            Permission.READ_RESULTS,
        },
    }
    
    @classmethod
    def get_role_permissions(cls, role: Role) -> Set[Permission]:
        """Get all permissions for a role"""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def has_permission(cls, role: Role, permission: Permission) -> bool:
        """Check if a role has a specific permission"""
        return permission in cls.get_role_permissions(role)
    
    @classmethod
    def get_role_display_name(cls, role: Role) -> str:
        """Get human-readable name for role"""
        names = {
            Role.ADMIN: "Administrator",
            Role.USER: "User",
            Role.VIEWER: "Viewer",
            Role.SERVICE: "Service",
        }
        return names.get(role, role.value)
    
    @classmethod
    def _build_rbac_config(cls) -> Dict[str, Any]:
        """Build full RBAC configuration for deployment"""
        return {
            "version": "1.0",
            "roles": {
                role.value: {
                    "display_name": cls.get_role_display_name(role),
                    "permissions": [p.value for p in permissions]
                }
                for role, permissions in cls.ROLE_PERMISSIONS.items()
            },
            "permission_hierarchy": {
                "admin": ["user", "viewer", "service"],
                "user": ["viewer"],
                "viewer": [],
                "service": []
            }
        }


@dataclass
class UserContext:
    """Container for user identity and role information"""
    user_id: str
    email: str
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    custom_claims: Dict[str, Any] = field(default_factory=dict)
    issued_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_token_claims(cls, claims: Dict[str, Any]) -> "UserContext":
        """Create UserContext from Firebase token claims"""
        role_str = claims.get("role", "viewer")
        role = Role(role_str) if role_str in [r.value for r in Role] else Role.VIEWER
        
        permissions = RBACConfig.get_role_permissions(role)
        
        return cls(
            user_id=claims.get("sub", ""),
            email=claims.get("email", ""),
            role=role,
            permissions=permissions,
            custom_claims=claims,
        )
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the given permissions"""
        return any(self.has_permission(p) for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the given permissions"""
        return all(self.has_permission(p) for p in permissions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "issued_at": self.issued_at.isoformat(),
        }


class RBACManager:
    """Manager for RBAC operations"""
    
    def __init__(self):
        """Initialize RBAC manager"""
        self.config = RBACConfig()
        self._audit_log: List[Dict[str, Any]] = []
    
    def authorize(self, context: UserContext, required_permission: Permission) -> bool:
        """
        Check if user has required permission
        
        Args:
            context: User context with role info
            required_permission: Permission to check
            
        Returns:
            True if authorized, False otherwise
        """
        authorized = context.has_permission(required_permission)
        
        # Log access attempt
        self._log_access(
            user_id=context.user_id,
            permission=required_permission,
            granted=authorized,
        )
        
        return authorized
    
    def authorize_any(self, context: UserContext, permissions: List[Permission]) -> bool:
        """Check if user has any of the required permissions"""
        return context.has_any_permission(permissions)
    
    def authorize_all(self, context: UserContext, permissions: List[Permission]) -> bool:
        """Check if user has all of the required permissions"""
        return context.has_all_permissions(permissions)
    
    def get_user_capabilities(self, context: UserContext) -> Dict[str, Any]:
        """Get list of capabilities for user's role"""
        return {
            "role": context.role.value,
            "permissions": [p.value for p in context.permissions],
            "role_name": RBACConfig.get_role_display_name(context.role),
        }
    
    def _log_access(self, user_id: str, permission: Permission, granted: bool):
        """Log access attempt for audit trail"""
        self._audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "permission": permission.value,
            "granted": granted,
        })
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get access audit log"""
        return self._audit_log.copy()
    
    def clear_audit_log(self):
        """Clear audit log"""
        self._audit_log.clear()


# Singleton instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create RBAC manager singleton"""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager

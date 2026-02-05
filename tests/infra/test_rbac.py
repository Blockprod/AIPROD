"""
Comprehensive tests for RBAC implementation
Tests role definitions, permissions, user context, and access control
"""

import pytest
from datetime import datetime
from src.infra.rbac import (
    Role, Permission, RBACConfig, UserContext, RBACManager, get_rbac_manager
)


class TestRoleDefinitions:
    """Test role and permission definitions"""
    
    def test_roles_exist(self):
        """Test that all expected roles are defined"""
        expected_roles = {Role.ADMIN, Role.USER, Role.VIEWER, Role.SERVICE}
        actual_roles = set(Role)
        assert actual_roles == expected_roles
    
    def test_permissions_exist(self):
        """Test that all expected permissions are defined"""
        permissions = set(Permission)
        assert len(permissions) >= 12
        assert Permission.READ_JOBS in permissions
        assert Permission.CREATE_JOB in permissions
        assert Permission.MANAGE_USERS in permissions


class TestRBACConfiguration:
    """Test RBAC configuration and permission matrix"""
    
    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions"""
        admin_perms = RBACConfig.get_role_permissions(Role.ADMIN)
        all_perms = set(Permission)
        assert all_perms.issubset(admin_perms)
    
    def test_user_has_expected_permissions(self):
        """Test that user role has expected permissions"""
        user_perms = RBACConfig.get_role_permissions(Role.USER)
        expected = {
            Permission.READ_JOBS,
            Permission.CREATE_JOB,
            Permission.UPDATE_JOB,
            Permission.DELETE_JOB,
        }
        assert expected.issubset(user_perms)
        assert Permission.MANAGE_USERS not in user_perms
    
    def test_viewer_has_read_only_permissions(self):
        """Test that viewer role has only read permissions"""
        viewer_perms = RBACConfig.get_role_permissions(Role.VIEWER)
        expected = {
            Permission.READ_JOBS,
            Permission.READ_RESULTS,
            Permission.READ_LOGS,
            Permission.READ_METRICS,
        }
        assert expected == viewer_perms
    
    def test_service_role_permissions(self):
        """Test that service role has service permissions"""
        service_perms = RBACConfig.get_role_permissions(Role.SERVICE)
        assert Permission.SERVICE_CALL in service_perms
        assert Permission.SERVICE_WEBHOOK in service_perms
    
    def test_has_permission(self):
        """Test permission checking"""
        assert RBACConfig.has_permission(Role.ADMIN, Permission.READ_JOBS)
        assert RBACConfig.has_permission(Role.USER, Permission.CREATE_JOB)
        assert not RBACConfig.has_permission(Role.VIEWER, Permission.DELETE_JOB)
    
    def test_role_display_names(self):
        """Test role display names"""
        assert RBACConfig.get_role_display_name(Role.ADMIN) == "Administrator"
        assert RBACConfig.get_role_display_name(Role.USER) == "User"
        assert RBACConfig.get_role_display_name(Role.VIEWER) == "Viewer"
        assert RBACConfig.get_role_display_name(Role.SERVICE) == "Service"


class TestUserContext:
    """Test UserContext class"""
    
    def test_user_context_creation(self):
        """Test creating user context"""
        user = UserContext(
            user_id="user123",
            email="user@example.com",
            role=Role.USER
        )
        assert user.user_id == "user123"
        assert user.email == "user@example.com"
        assert user.role == Role.USER
    
    def test_user_context_permissions_populated(self):
        """Test that user context gets role permissions"""
        user = UserContext(
            user_id="user123",
            email="user@example.com",
            role=Role.USER
        )
        # Permissions should be empty initially (set by manager)
        assert len(user.permissions) == 0
    
    def test_user_context_from_token_claims_admin(self):
        """Test creating user context from admin claims"""
        claims = {
            "sub": "admin123",
            "email": "admin@example.com",
            "role": "admin"
        }
        user = UserContext.from_token_claims(claims)
        assert user.user_id == "admin123"
        assert user.email == "admin@example.com"
        assert user.role == Role.ADMIN
        assert len(user.permissions) > 0
    
    def test_user_context_from_token_claims_viewer(self):
        """Test creating user context from viewer claims"""
        claims = {
            "sub": "viewer123",
            "email": "viewer@example.com",
            "role": "viewer"
        }
        user = UserContext.from_token_claims(claims)
        assert user.role == Role.VIEWER
        assert Permission.READ_JOBS in user.permissions
        assert Permission.CREATE_JOB not in user.permissions
    
    def test_user_context_from_invalid_claims_defaults_to_viewer(self):
        """Test that invalid role defaults to viewer"""
        claims = {
            "sub": "user123",
            "email": "user@example.com",
            "role": "invalid_role"
        }
        user = UserContext.from_token_claims(claims)
        assert user.role == Role.VIEWER
    
    def test_has_permission(self):
        """Test user permission checking"""
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        assert user.has_permission(Permission.CREATE_JOB)
        assert not user.has_permission(Permission.MANAGE_USERS)
    
    def test_has_any_permission(self):
        """Test checking for any of multiple permissions"""
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        assert user.has_any_permission([Permission.CREATE_JOB, Permission.MANAGE_USERS])
        assert not user.has_any_permission([Permission.MANAGE_USERS, Permission.MANAGE_ROLES])
    
    def test_has_all_permissions(self):
        """Test checking for all required permissions"""
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        assert user.has_all_permissions([Permission.CREATE_JOB, Permission.READ_JOBS])
        assert not user.has_all_permissions([Permission.CREATE_JOB, Permission.MANAGE_USERS])
    
    def test_user_context_to_dict(self):
        """Test converting user context to dict"""
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        user_dict = user.to_dict()
        assert user_dict["user_id"] == "user123"
        assert user_dict["email"] == "user@example.com"
        assert user_dict["role"] == "user"
        assert "permissions" in user_dict
        assert "issued_at" in user_dict


class TestRBACManager:
    """Test RBACManager class"""
    
    def test_rbac_manager_creation(self):
        """Test creating RBAC manager"""
        manager = RBACManager()
        assert manager is not None
        assert manager.config is not None
    
    def test_authorize_with_permission(self):
        """Test authorization when user has permission"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        assert manager.authorize(user, Permission.CREATE_JOB)
    
    def test_authorize_without_permission(self):
        """Test authorization when user lacks permission"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "viewer123",
            "email": "viewer@example.com",
            "role": "viewer"
        })
        assert not manager.authorize(user, Permission.DELETE_JOB)
    
    def test_authorize_any(self):
        """Test checking for any permission"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        assert manager.authorize_any(user, [Permission.DELETE_JOB, Permission.CREATE_JOB])
        assert not manager.authorize_any(user, [Permission.MANAGE_USERS, Permission.MANAGE_ROLES])
    
    def test_authorize_all(self):
        """Test checking for all permissions"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        assert manager.authorize_all(user, [Permission.CREATE_JOB, Permission.READ_JOBS])
        assert not manager.authorize_all(user, [Permission.CREATE_JOB, Permission.MANAGE_USERS])
    
    def test_get_user_capabilities(self):
        """Test getting user capabilities"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "admin123",
            "email": "admin@example.com",
            "role": "admin"
        })
        caps = manager.get_user_capabilities(user)
        assert caps["role"] == "admin"
        assert "permissions" in caps
        assert "role_name" in caps
    
    def test_audit_logging(self):
        """Test audit log for access attempts"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        
        manager.authorize(user, Permission.CREATE_JOB)
        manager.authorize(user, Permission.MANAGE_USERS)
        
        log = manager.get_audit_log()
        assert len(log) == 2
        assert log[0]["user_id"] == "user123"
        assert log[0]["permission"] == Permission.CREATE_JOB.value
        assert log[0]["granted"] is True
        assert log[1]["granted"] is False
    
    def test_clear_audit_log(self):
        """Test clearing audit log"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        manager.authorize(user, Permission.CREATE_JOB)
        assert len(manager.get_audit_log()) > 0
        manager.clear_audit_log()
        assert len(manager.get_audit_log()) == 0


class TestRBACIntegration:
    """Integration tests for RBAC system"""
    
    def test_complete_permission_flow_admin(self):
        """Test complete flow for admin access"""
        manager = RBACManager()
        admin = UserContext.from_token_claims({
            "sub": "admin123",
            "email": "admin@example.com",
            "role": "admin"
        })
        
        # Admin should have all permissions
        assert manager.authorize(admin, Permission.CREATE_JOB)
        assert manager.authorize(admin, Permission.MANAGE_USERS)
        assert manager.authorize(admin, Permission.DELETE_JOB)
    
    def test_complete_permission_flow_user(self):
        """Test complete flow for regular user"""
        manager = RBACManager()
        user = UserContext.from_token_claims({
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        })
        
        # User should have job management but not admin
        assert manager.authorize(user, Permission.CREATE_JOB)
        assert manager.authorize(user, Permission.DELETE_JOB)
        assert not manager.authorize(user, Permission.MANAGE_USERS)
    
    def test_complete_permission_flow_viewer(self):
        """Test complete flow for viewer"""
        manager = RBACManager()
        viewer = UserContext.from_token_claims({
            "sub": "viewer123",
            "email": "viewer@example.com",
            "role": "viewer"
        })
        
        # Viewer should only read
        assert manager.authorize(viewer, Permission.READ_JOBS)
        assert not manager.authorize(viewer, Permission.CREATE_JOB)
        assert not manager.authorize(viewer, Permission.MANAGE_USERS)
    
    def test_rbac_singleton(self):
        """Test RBAC manager singleton pattern"""
        manager1 = get_rbac_manager()
        manager2 = get_rbac_manager()
        assert manager1 is manager2
    
    def test_role_hierarchy(self):
        """Test role hierarchy is respected"""
        admin = UserContext.from_token_claims({
            "sub": "admin",
            "email": "admin@example.com",
            "role": "admin"
        })
        user = UserContext.from_token_claims({
            "sub": "user",
            "email": "user@example.com",
            "role": "user"
        })
        
        # Admin should have all user permissions
        assert all(
            admin.has_permission(p) for p in RBACConfig.get_role_permissions(Role.USER)
        )

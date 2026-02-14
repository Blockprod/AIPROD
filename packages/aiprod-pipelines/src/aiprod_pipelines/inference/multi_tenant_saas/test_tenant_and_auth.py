"""
Comprehensive tests for tenant management and authentication.

Tests:
  - TenantMetadata and TenantContext
  - TenantRegistry and context management
  - API keys and JWT tokens
  - JWT token manager
  - Session management
  - Authentication manager
"""

import unittest
from datetime import datetime, timedelta
import time

from aiprod_pipelines.inference.multi_tenant_saas.tenant_context import (
    TenantMetadata,
    TenantContext,
    TenantContextManager,
    TenantRegistry,
    TenantTier,
    TenantStatus,
    TenantQuotas,
    get_current_tenant_context,
)

from aiprod_pipelines.inference.multi_tenant_saas.authentication import (
    APIKey,
    JWTTokenManager,
    SessionManager,
    AuthenticationManager,
    TokenType,
    AuthenticationMethod,
)


class TestTenantMetadata(unittest.TestCase):
    """Test tenant metadata."""
    
    def test_create_tenant(self):
        """Test tenant creation."""
        tenant = TenantMetadata(
            tenant_id="tenant_1",
            organization_name="Test Org",
            tier=TenantTier.PROFESSIONAL,
        )
        self.assertEqual(tenant.tenant_id, "tenant_1")
        self.assertEqual(tenant.organization_name, "Test Org")
        self.assertTrue(tenant.is_active())
    
    def test_tenant_quotas(self):
        """Test tenant quotas."""
        quotas = TenantQuotas(max_concurrent_jobs=10, max_videos_per_month=200)
        self.assertEqual(quotas.max_concurrent_jobs, 10)
        self.assertEqual(quotas.max_videos_per_month, 200)
    
    def test_tenant_features(self):
        """Test feature management."""
        tenant = TenantMetadata(tenant_id="t1", organization_name="Org")
        tenant.features.add("lora")
        tenant.features.add("quantization")
        
        self.assertTrue(tenant.has_feature("lora"))
        self.assertTrue(tenant.has_feature("quantization"))
        self.assertFalse(tenant.has_feature("distillation"))
    
    def test_trial_period(self):
        """Test trial period tracking."""
        future_date = datetime.utcnow() + timedelta(days=14)
        tenant = TenantMetadata(
            tenant_id="t1",
            organization_name="Org",
            status=TenantStatus.TRIAL,
            subscription_end_date=future_date,
        )
        
        days_left = tenant.trial_remaining_days()
        self.assertIsNotNone(days_left)
        self.assertGreater(days_left, 0)
        self.assertLessEqual(days_left, 14)


class TestTenantContext(unittest.TestCase):
    """Test tenant context."""
    
    def test_create_context(self):
        """Test context creation."""
        metadata = TenantMetadata(tenant_id="t1", organization_name="Org")
        context = TenantContext(
            tenant_id="t1",
            tenant_metadata=metadata,
            request_id="req_1",
            user_id="user_1",
        )
        
        self.assertEqual(context.get_tenant_id(), "t1")
        self.assertEqual(context.user_id, "user_1")
    
    def test_custom_data(self):
        """Test custom request data."""
        metadata = TenantMetadata(tenant_id="t1", organization_name="Org")
        context = TenantContext(
            tenant_id="t1",
            tenant_metadata=metadata,
            request_id="req_1",
        )
        
        context.set_custom_data("model", "ltx-video-2")
        self.assertEqual(context.get_custom_data("model"), "ltx-video-2")
        self.assertIsNone(context.get_custom_data("nonexistent"))


class TestTenantRegistry(unittest.TestCase):
    """Test tenant registry."""
    
    def test_register_tenant(self):
        """Test tenant registration."""
        registry = TenantRegistry()
        tenant = TenantMetadata(tenant_id="t1", organization_name="Org")
        
        registry.register_tenant(tenant)
        retrieved = registry.get_tenant("t1")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.tenant_id, "t1")
    
    def test_update_tenant_status(self):
        """Test tenant status update."""
        registry = TenantRegistry()
        tenant = TenantMetadata(tenant_id="t1", organization_name="Org")
        registry.register_tenant(tenant)
        
        registry.update_tenant_status("t1", TenantStatus.SUSPENDED)
        tenant = registry.get_tenant("t1")
        
        self.assertEqual(tenant.status, TenantStatus.SUSPENDED)
        self.assertFalse(tenant.is_active())
    
    def test_add_remove_features(self):
        """Test adding and removing features."""
        registry = TenantRegistry()
        tenant = TenantMetadata(tenant_id="t1", organization_name="Org")
        registry.register_tenant(tenant)
        
        registry.add_feature("t1", "lora")
        tenant = registry.get_tenant("t1")
        self.assertTrue(tenant.has_feature("lora"))
        
        registry.remove_feature("t1", "lora")
        tenant = registry.get_tenant("t1")
        self.assertFalse(tenant.has_feature("lora"))
    
    def test_list_active_tenants(self):
        """Test listing active tenants."""
        registry = TenantRegistry()
        
        t1 = TenantMetadata(tenant_id="t1", organization_name="Org1")
        t2 = TenantMetadata(tenant_id="t2", organization_name="Org2", status=TenantStatus.SUSPENDED)
        
        registry.register_tenant(t1)
        registry.register_tenant(t2)
        
        active = registry.list_active_tenants()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].tenant_id, "t1")


class TestAPIKey(unittest.TestCase):
    """Test API key management."""
    
    def test_create_api_key(self):
        """Test API key creation."""
        auth_manager = AuthenticationManager("secret_key")
        key, api_key_obj = auth_manager.create_api_key("t1", "test_key")
        
        self.assertIsNotNone(key)
        self.assertTrue(api_key_obj.is_valid())
    
    def test_verify_api_key(self):
        """Test API key verification."""
        auth_manager = AuthenticationManager("secret_key")
        key, _ = auth_manager.create_api_key("t1")
        
        verified = auth_manager.verify_api_key(key)
        self.assertIsNotNone(verified)
        self.assertTrue(verified.is_valid())
    
    def test_revoke_api_key(self):
        """Test API key revocation."""
        auth_manager = AuthenticationManager("secret_key")
        key, api_key_obj = auth_manager.create_api_key("t1")
        key_hash = api_key_obj.key_hash
        
        auth_manager.revoke_api_key(key_hash)
        verified = auth_manager.verify_api_key(key)
        self.assertIsNone(verified)
    
    def test_api_key_expiration(self):
        """Test API key expiration."""
        auth_manager = AuthenticationManager("secret_key")
        _, api_key_obj = auth_manager.create_api_key("t1", expires_in_days=0)
        
        # Manually set past expiration
        api_key_obj.expires_at = datetime.utcnow() - timedelta(days=1)
        
        self.assertTrue(api_key_obj.is_expired())
        self.assertFalse(api_key_obj.is_valid())


class TestJWTTokenManager(unittest.TestCase):
    """Test JWT token manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.jwt_manager = JWTTokenManager("test_secret_key")
    
    def test_generate_token(self):
        """Test token generation."""
        token, token_data = self.jwt_manager.generate_token(
            tenant_id="t1",
            user_id="user_1",
        )
        
        self.assertIsNotNone(token)
        self.assertIsNotNone(token_data.token_id)
        self.assertEqual(token_data.tenant_id, "t1")
        self.assertEqual(token_data.user_id, "user_1")
    
    def test_verify_token(self):
        """Test token verification."""
        token, _ = self.jwt_manager.generate_token("t1", "user_1")
        verified = self.jwt_manager.verify_token(token)
        
        self.assertIsNotNone(verified)
        self.assertEqual(verified.tenant_id, "t1")
        self.assertEqual(verified.user_id, "user_1")
    
    def test_revoke_token(self):
        """Test token revocation."""
        token, token_data = self.jwt_manager.generate_token("t1", "user_1")
        
        self.jwt_manager.revoke_token(token_data.token_id)
        verified = self.jwt_manager.verify_token(token)
        
        self.assertIsNone(verified)
    
    def test_token_expiration(self):
        """Test token expiration."""
        token, _ = self.jwt_manager.generate_token("t1", "user_1", expires_in_hours=0)
        
        # Wait a bit and verify token is expired
        time.sleep(0.1)
        verified = self.jwt_manager.verify_token(token)
        
        self.assertIsNone(verified)
    
    def test_refresh_token(self):
        """Test refresh token generation."""
        token, token_data = self.jwt_manager.generate_refresh_token("t1", "user_1")
        
        self.assertTrue(token_data.is_refresh_token)
        self.assertIn("refresh", token_data.scopes)


class TestSessionManager(unittest.TestCase):
    """Test session management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager(session_timeout_minutes=1)
    
    def test_create_session(self):
        """Test session creation."""
        session_id = self.session_manager.create_session("t1", "user_1")
        
        self.assertIsNotNone(session_id)
        self.assertTrue(self.session_manager.validate_session(session_id))
    
    def test_session_info(self):
        """Test retrieving session info."""
        session_id = self.session_manager.create_session("t1", "user_1")
        info = self.session_manager.get_session_info(session_id)
        
        self.assertIsNotNone(info)
        self.assertEqual(info["tenant_id"], "t1")
        self.assertEqual(info["user_id"], "user_1")
    
    def test_session_timeout(self):
        """Test session timeout."""
        # Use very short timeout
        sm = SessionManager(session_timeout_minutes=0)
        session_id = sm.create_session("t1", "user_1")
        
        # Sessions with 0 minute timeout should timeout immediately
        self.assertFalse(sm.validate_session(session_id))
    
    def test_revoke_session(self):
        """Test session revocation."""
        session_id = self.session_manager.create_session("t1", "user_1")
        
        self.session_manager.revoke_session(session_id)
        self.assertFalse(self.session_manager.validate_session(session_id))
    
    def test_revoke_user_sessions(self):
        """Test revoking all user sessions."""
        sid1 = self.session_manager.create_session("t1", "user_1")
        sid2 = self.session_manager.create_session("t1", "user_1")
        sid3 = self.session_manager.create_session("t1", "user_2")
        
        revoked = self.session_manager.revoke_user_sessions("t1", "user_1")
        
        self.assertEqual(revoked, 2)
        self.assertFalse(self.session_manager.validate_session(sid1))
        self.assertFalse(self.session_manager.validate_session(sid2))
        self.assertTrue(self.session_manager.validate_session(sid3))


if __name__ == "__main__":
    unittest.main()

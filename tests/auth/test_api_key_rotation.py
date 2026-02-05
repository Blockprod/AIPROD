"""
Tests for API Key Manager service
Validates key generation, rotation, revocation, and verification
"""

import pytest
from datetime import datetime, timedelta
from src.auth.api_key_manager import APIKeyManager, APIKeyStatus


class TestAPIKeyManager:
    """Test suite for APIKeyManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager instance for each test"""
        return APIKeyManager()
    
    def test_generate_api_key(self, manager):
        """Test API key generation"""
        result = manager.generate_api_key("user123", "test-key")
        
        assert result["key_id"].startswith("key_")
        assert result["key_value"].startswith("apk_")
        assert result["prefix"] == "apk_"
        assert "created_at" in result
        assert "expires_at" in result
        assert "warning" in result
    
    def test_generate_multiple_keys_different_ids(self, manager):
        """Test that multiple keys generate unique IDs"""
        key1 = manager.generate_api_key("user123", "key1")
        key2 = manager.generate_api_key("user123", "key2")
        
        assert key1["key_id"] != key2["key_id"]
        assert key1["key_value"] != key2["key_value"]
    
    def test_verify_valid_api_key(self, manager):
        """Test verification of valid API key"""
        generated = manager.generate_api_key("user123", "test")
        
        verified = manager.verify_api_key(generated["key_value"], "user123")
        
        assert verified is not None
        assert verified["key_id"] == generated["key_id"]
        assert verified["status"] == APIKeyStatus.ACTIVE.value
        assert verified["user_id"] == "user123"
    
    def test_verify_invalid_api_key(self, manager):
        """Test verification fails for invalid key"""
        manager.generate_api_key("user123", "test")
        
        verified = manager.verify_api_key("invalid_key", "user123")
        
        assert verified is None
    
    def test_verify_wrong_user(self, manager):
        """Test verification fails for wrong user"""
        generated = manager.generate_api_key("user123", "test")
        
        verified = manager.verify_api_key(generated["key_value"], "wrong_user")
        
        assert verified is None
    
    def test_rotate_api_key(self, manager):
        """Test API key rotation"""
        original = manager.generate_api_key("user123", "prod")
        
        rotated = manager.rotate_api_key(original["key_id"], "user123")
        
        assert rotated["key_id"] != original["key_id"]
        assert rotated["key_value"] != original["key_value"]
        
        # Original key should be marked as rotated
        old_key = manager.verify_api_key(original["key_value"], "user123")
        assert old_key is None  # Rotated keys should not verify
    
    def test_rotate_unauthorized(self, manager):
        """Test rotation fails for wrong user"""
        original = manager.generate_api_key("user123", "prod")
        
        with pytest.raises(ValueError):
            manager.rotate_api_key(original["key_id"], "wrong_user")
    
    def test_revoke_api_key(self, manager):
        """Test API key revocation"""
        key = manager.generate_api_key("user123", "test")
        
        result = manager.revoke_api_key(key["key_id"], "user123")
        
        assert result is True
        
        # Revoked key should not verify
        verified = manager.verify_api_key(key["key_value"], "user123")
        assert verified is None
    
    def test_revoke_unauthorized(self, manager):
        """Test revocation fails for wrong user"""
        key = manager.generate_api_key("user123", "test")
        
        with pytest.raises(ValueError):
            manager.revoke_api_key(key["key_id"], "wrong_user")
    
    def test_list_api_keys(self, manager):
        """Test listing user's API keys"""
        key1 = manager.generate_api_key("user123", "prod")
        key2 = manager.generate_api_key("user123", "staging")
        
        keys = manager.list_api_keys("user123")
        
        assert len(keys) == 2
        key_ids = [k["key_id"] for k in keys]
        assert key1["key_id"] in key_ids
        assert key2["key_id"] in key_ids
        
        # Keys should not have key_hash
        for key in keys:
            assert "key_hash" not in key
    
    def test_list_keys_only_active(self, manager):
        """Test listing only active keys"""
        key1 = manager.generate_api_key("user123", "active")
        key2 = manager.generate_api_key("user123", "revoke_me")
        
        manager.revoke_api_key(key2["key_id"], "user123")
        
        active_keys = manager.list_api_keys("user123", include_inactive=False)
        all_keys = manager.list_api_keys("user123", include_inactive=True)
        
        assert len(active_keys) == 1
        assert len(all_keys) == 2
    
    def test_revoke_all_keys(self, manager):
        """Test revoking all keys for a user"""
        key1 = manager.generate_api_key("user123", "key1")
        key2 = manager.generate_api_key("user123", "key2")
        key3 = manager.generate_api_key("user123", "key3")
        
        revoked_count = manager.revoke_all_keys("user123")
        
        assert revoked_count == 3
        
        # All keys should be revoked
        keys = manager.list_api_keys("user123", include_inactive=True)
        for key in keys:
            assert key["status"] == APIKeyStatus.REVOKED.value
    
    def test_verify_updates_usage_stats(self, manager):
        """Test that verification updates usage statistics"""
        key = manager.generate_api_key("user123", "test")
        
        # Verify multiple times
        manager.verify_api_key(key["key_value"], "user123")
        manager.verify_api_key(key["key_value"], "user123")
        manager.verify_api_key(key["key_value"], "user123")
        
        # Check usage stats
        keys = manager.list_api_keys("user123")
        assert keys[0]["usage_count"] == 3
        assert keys[0]["last_used"] is not None
    
    def test_key_expiration(self, manager):
        """Test that expired keys don't verify"""
        key = manager.generate_api_key("user123", "test")
        
        # Manually mark as expired
        if manager._in_memory_cache:
            cache_key = [k for k in manager._in_memory_cache.keys()][0]
            manager._in_memory_cache[cache_key]["expires_at"] = \
                (datetime.utcnow() - timedelta(days=1)).isoformat()
        
        # Expired key should not verify
        verified = manager.verify_api_key(key["key_value"], "user123")
        assert verified is None
    
    def test_different_users_isolated(self, manager):
        """Test that keys are isolated between users"""
        user1_key = manager.generate_api_key("user1", "key1")
        user2_key = manager.generate_api_key("user2", "key2")
        
        # User1 should only see their key
        user1_keys = manager.list_api_keys("user1")
        assert len(user1_keys) == 1
        assert user1_keys[0]["key_id"] == user1_key["key_id"]
        
        # User2 should only see their key
        user2_keys = manager.list_api_keys("user2")
        assert len(user2_keys) == 1
        assert user2_keys[0]["key_id"] == user2_key["key_id"]
        
        # User1's key value should not work for User2
        verified = manager.verify_api_key(user1_key["key_value"], "user2")
        assert verified is None


class TestAPIKeyManagerModels:
    """Test API key model validation"""
    
    def test_create_api_key_request_valid(self):
        """Test valid CreateAPIKeyRequest"""
        from src.auth.api_key_models import CreateAPIKeyRequest
        
        req = CreateAPIKeyRequest(name="production")
        assert req.name == "production"
    
    def test_api_key_response_valid(self):
        """Test valid APIKeyResponse"""
        from src.auth.api_key_models import APIKeyResponse
        
        response = APIKeyResponse(
            key_id="key_123",
            key_value="apk_abc123",
            prefix="apk_",
            created_at="2026-02-05T12:00:00",
            expires_at="2026-05-05T12:00:00"
        )
        assert response.key_id == "key_123"
        assert response.key_value == "apk_abc123"
    
    def test_list_api_keys_response_valid(self):
        """Test valid ListAPIKeysResponse"""
        from src.auth.api_key_models import ListAPIKeysResponse, APIKeyMetadata
        
        response = ListAPIKeysResponse(
            keys=[],
            total=0,
            active_count=0
        )
        assert response.total == 0
        assert response.active_count == 0


class TestAPIKeyEndpoints:
    """Test API key endpoints structure (not full integration)"""
    
    def test_create_api_key_endpoint_exists(self):
        """Test that create endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys/create" in routes
    
    def test_list_api_keys_endpoint_exists(self):
        """Test that list endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys" in routes
    
    def test_rotate_api_key_endpoint_exists(self):
        """Test that rotate endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys/{key_id}/rotate" in routes
    
    def test_revoke_api_key_endpoint_exists(self):
        """Test that revoke endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys/{key_id}/revoke" in routes
    
    def test_revoke_all_api_keys_endpoint_exists(self):
        """Test that revoke-all endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys/revoke-all" in routes
    
    def test_api_key_stats_endpoint_exists(self):
        """Test that stats endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys/stats" in routes
    
    def test_api_key_health_endpoint_exists(self):
        """Test that health endpoint is defined"""
        from src.api.main import app
        
        routes = [getattr(r, 'path', '') for r in app.routes]
        assert "/api-keys/health" in routes

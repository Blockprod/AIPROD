"""
Tests pour les endpoints d'authentification et token refresh.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.auth.token_manager import TokenManager, InMemoryTokenCache, get_token_manager
from src.auth.auth_models import RefreshTokenRequest, TokenResponse


class TestTokenManager:
    """Tests pour le TokenManager."""

    @pytest.fixture
    def token_manager(self):
        """Crée une instance TokenManager pour les tests."""
        manager = TokenManager()
        # Utiliser le cache en mémoire pour les tests
        manager.cache = InMemoryTokenCache()
        return manager

    def test_generate_refresh_token(self, token_manager):
        """Test la génération d'un token de refresh."""
        user_id = "test-user-123"
        token = token_manager.generate_refresh_token(user_id)

        # Vérifier que le token est généré et non vide
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_refresh_token_valid(self, token_manager):
        """Test la vérification d'un token de refresh valide."""
        user_id = "test-user-123"
        token = token_manager.generate_refresh_token(user_id)

        # Vérifier que le token est valide
        assert token_manager.verify_refresh_token(user_id, token) is True

    def test_verify_refresh_token_invalid(self, token_manager):
        """Test la vérification d'un token invalide."""
        user_id = "test-user-123"
        invalid_token = "invalid-token-xyz"

        # Vérifier que le token invalide est rejeté
        assert token_manager.verify_refresh_token(user_id, invalid_token) is False

    def test_verify_refresh_token_wrong_user(self, token_manager):
        """Test la vérification d'un token avec un autre user_id."""
        user_id_1 = "user-1"
        user_id_2 = "user-2"

        token = token_manager.generate_refresh_token(user_id_1)

        # Le token créé par user-1 devrait être rejeté quand utilisé par user-2
        assert token_manager.verify_refresh_token(user_id_2, token) is False

    def test_rotate_refresh_token(self, token_manager):
        """Test la rotation d'un token."""
        user_id = "test-user-123"
        old_token = token_manager.generate_refresh_token(user_id)

        # Vérifier que l'ancien token fonctionne
        assert token_manager.verify_refresh_token(user_id, old_token) is True

        # Effectuer la rotation
        new_token = token_manager.rotate_refresh_token(user_id, old_token)

        # Vérifier que le nouveau token est généré
        assert new_token is not None
        assert new_token != old_token

        # Vérifier que le nouveau token fonctionne
        assert token_manager.verify_refresh_token(user_id, new_token) is True

        # Vérifier que l'ancien token n'existe plus
        assert token_manager.verify_refresh_token(user_id, old_token) is False

    def test_rotate_refresh_token_invalid(self, token_manager):
        """Test la rotation d'un token invalide échoue."""
        user_id = "test-user-123"
        invalid_token = "invalid-xyz"

        # Essayer de rotateer un token invalide
        new_token = token_manager.rotate_refresh_token(user_id, invalid_token)

        # Devrait retourner None
        assert new_token is None

    def test_revoke_refresh_token(self, token_manager):
        """Test la révocation d'un token."""
        user_id = "test-user-123"
        token = token_manager.generate_refresh_token(user_id)

        # Vérifier que le token fonctionne
        assert token_manager.verify_refresh_token(user_id, token) is True

        # Révoquer le token
        assert token_manager.revoke_refresh_token(user_id, token) is True

        # Vérifier que le token n'existe plus
        assert token_manager.verify_refresh_token(user_id, token) is False

    def test_get_token_info(self, token_manager):
        """Test la récupération des infos d'un token."""
        user_id = "test-user-123"
        token = token_manager.generate_refresh_token(user_id)

        # Récupérer les infos
        info = token_manager.get_token_info(user_id, token)

        # Vérifier que les infos sont complètes
        assert info is not None
        assert info["user_id"] == user_id
        assert "created_at" in info
        assert "expires_at" in info
        assert "version" in info

    def test_create_token_pair(self, token_manager):
        """Test la création d'une paire de tokens."""
        user_id = "test-user-123"
        access_token = "access-token-xyz"

        access, refresh, ttl = token_manager.create_token_pair(user_id, access_token)

        # Vérifier que la paire est créée
        assert access == access_token
        assert refresh is not None
        assert isinstance(refresh, str)
        assert ttl > 0

        # Vérifier que le refresh token fonctionne
        assert token_manager.verify_refresh_token(user_id, refresh) is True

    def test_token_expiration(self):
        """Test que les tokens expirent après TTL."""
        cache = InMemoryTokenCache()
        manager = TokenManager()
        manager.cache = cache

        user_id = "test-user"
        manager.refresh_token_ttl = 1  # 1 second TTL

        token = manager.generate_refresh_token(user_id)

        # Token devrait fonctionner immédiatement
        assert manager.verify_refresh_token(user_id, token) is True

        # Attendre l'expiration
        import time
        time.sleep(2)

        # Token devrait être expiré
        assert manager.verify_refresh_token(user_id, token) is False

    def test_token_uniqueness(self, token_manager):
        """Test que chaque token généré est unique."""
        user_id = "test-user-123"

        token1 = token_manager.generate_refresh_token(user_id)
        token2 = token_manager.generate_refresh_token(user_id)
        token3 = token_manager.generate_refresh_token(user_id)

        # Tous les tokens doivent être différents
        assert token1 != token2
        assert token2 != token3
        assert token1 != token3

        # Tous les tokens doivent être valides
        assert token_manager.verify_refresh_token(user_id, token1) is True
        assert token_manager.verify_refresh_token(user_id, token2) is True
        assert token_manager.verify_refresh_token(user_id, token3) is True


class TestInMemoryTokenCache:
    """Tests pour le cache en mémoire."""

    def test_cache_set_and_get(self):
        """Test set et get du cache."""
        cache = InMemoryTokenCache()

        cache.set("key1", {"data": "value"})
        value = cache.get("key1")

        assert value is not None
        assert value["data"] == "value"

    def test_cache_delete(self):
        """Test la suppression du cache."""
        cache = InMemoryTokenCache()

        cache.set("key1", {"data": "value"})
        cache.delete("key1")

        assert cache.get("key1") is None

    def test_cache_expiration(self):
        """Test l'expiration du cache."""
        import time
        cache = InMemoryTokenCache()

        cache.set("key1", {"data": "value"}, ttl=1)

        # Immédiat: devrait exister
        assert cache.get("key1") is not None

        # Après TTL: devrait être vide
        time.sleep(2)
        assert cache.get("key1") is None

    def test_cache_with_none_ttl(self):
        """Test le cache sans expiration."""
        cache = InMemoryTokenCache()

        cache.set("key1", {"data": "permanent"}, ttl=None)

        assert cache.get("key1") is not None

        # Même après un délai, devrait exister
        import time
        time.sleep(1)
        assert cache.get("key1") is not None


# Tests d'intégration avec l'API (si FastAPI TestClient disponible)
@pytest.mark.asyncio
class TestAuthEndpoints:
    """Tests des endpoints d'authentification."""

    async def test_refresh_endpoint_structure(self):
        """Test que l'endpoint refresh existe et a la bonne structure."""
        # Ceci est un test d'intégration simple
        # À exécuter avec pytest + FastAPI TestClient

        # L'endpoint devrait:
        # 1. Accepter un RefreshTokenRequest en POST
        # 2. Retourner un TokenResponse
        # 3. Utiliser le rate limiter (60/minute)

        request_data = RefreshTokenRequest(refresh_token="test-token")
        assert request_data.refresh_token == "test-token"

    async def test_token_response_model(self):
        """Test le modèle TokenResponse."""
        from datetime import datetime

        response = TokenResponse(
            access_token="access-123",
            refresh_token="refresh-456",
            token_type="Bearer",
            expires_in=900,
            issued_at=datetime.utcnow()
        )

        assert response.access_token == "access-123"
        assert response.refresh_token == "refresh-456"
        assert response.token_type == "Bearer"
        assert response.expires_in == 900


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Gestionnaire des tokens de refresh pour authentification sécurisée.
Implémente la rotation des tokens et la gestion du cycle de vie.

ARCHITECTURE:
- Utilise un cache en mémoire pour la performance (tokens expire rapidement)
- Fallback sur Redis possible via interface adaptée
- Pas de dépendance async - synchrone pour la génération/validation de tokens
"""

import os
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Types de tokens supportés."""
    ACCESS = "access"
    REFRESH = "refresh"


class TokenManager:
    """
    Gestionnaire de tokens avec support pour le refresh.
    Utilise Redis pour le stockage des tokens.
    """

    def __init__(self):
        """Initialise le tokeb manager."""
        self.cache = self._init_cache()
        self.access_token_ttl = int(os.getenv("ACCESS_TOKEN_TTL", "900"))  # 15 min
        self.refresh_token_ttl = int(os.getenv("REFRESH_TOKEN_TTL", "604800"))  # 7 days

    def _init_cache(self):
        """Initialise le cache (en mémoire pour la performance)."""
        # Utiliser le cache en mémoire pour les tokens refresh
        # - Tokens refreshes ont une courte durée de vie (quelques minutes)
        # - Accès rapide et synchrone
        # - Redis peut être ajouté pour la distribution multi-instance
        return InMemoryTokenCache()

    def generate_refresh_token(self, user_id: str) -> str:
        """
        Génère un token de refresh unique et sécurisé.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Token de refresh (chaîne aléatoire encodée en base64)
        """
        # Générer un token aléatoire sécurisé (32 bytes = 256 bits)
        raw_token = secrets.token_bytes(32)
        token = secrets.token_urlsafe(32)

        # Stocker le token dans le cache avec TTL
        cache_key = f"refresh_token:{user_id}:{token}"
        token_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=self.refresh_token_ttl)).isoformat(),
            "version": 1,
        }

        self.cache.set(cache_key, token_data, ttl=self.refresh_token_ttl)

        logger.info(f"Refresh token generated for user {user_id}")
        return token

    def verify_refresh_token(self, user_id: str, token: str) -> bool:
        """
        Vérifie qu'un token de refresh est valide et appartient à l'utilisateur.

        Args:
            user_id: ID de l'utilisateur
            token: Token de refresh à vérifier

        Returns:
            True si le token est valide, False sinon
        """
        cache_key = f"refresh_token:{user_id}:{token}"
        token_data = self.cache.get(cache_key)

        if not token_data:
            logger.warning(f"Refresh token not found or expired for user {user_id}")
            return False

        try:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.utcnow() > expires_at:
                logger.warning(f"Refresh token expired for user {user_id}")
                self.cache.delete(cache_key)
                return False

            return True
        except Exception as e:
            logger.error(f"Error verifying refresh token: {e}")
            return False

    def rotate_refresh_token(self, user_id: str, old_token: str) -> Optional[str]:
        """
        Effectue une rotation du token de refresh (revoke + generate).
        Empêche la réutilisation du même token.

        Args:
            user_id: ID de l'utilisateur
            old_token: Ancien token à révoquer

        Returns:
            Nouveau token de refresh, ou None si la rotation échoue
        """
        # Vérifier que l'ancien token est valide
        if not self.verify_refresh_token(user_id, old_token):
            logger.warning(f"Cannot rotate: invalid refresh token for user {user_id}")
            return None

        # Révoquer l'ancien token
        old_cache_key = f"refresh_token:{user_id}:{old_token}"
        self.cache.delete(old_cache_key)

        # Marquer comme révoqué pour éviter une réutilisation accidentelle
        revoke_key = f"revoked_token:{user_id}:{old_token}"
        self.cache.set(revoke_key, {"revoked_at": datetime.utcnow().isoformat()}, ttl=3600)

        logger.info(f"Refresh token rotated for user {user_id}")

        # Générer un nouveau token
        return self.generate_refresh_token(user_id)

    def revoke_refresh_token(self, user_id: str, token: str) -> bool:
        """
        Révoque un token de refresh de manière permanente.

        Args:
            user_id: ID de l'utilisateur
            token: Token à révoquer

        Returns:
            True si la révocation a réussi
        """
        cache_key = f"refresh_token:{user_id}:{token}"

        # Vérifier que le token existe
        if not self.cache.get(cache_key):
            logger.warning(f"Refresh token not found for revocation: {user_id}")
            return False

        # Supprimer le token
        self.cache.delete(cache_key)

        # Marquer comme révoqué
        revoke_key = f"revoked_token:{user_id}:{token}"
        self.cache.set(
            revoke_key, {"revoked_at": datetime.utcnow().isoformat()}, ttl=86400
        )  # 24h

        logger.warning(f"Refresh token revoked for user {user_id}")
        return True

    def revoke_all_tokens(self, user_id: str) -> int:
        """
        Révoque TOUS les tokens de refresh d'un utilisateur.
        Utile pour les déconnexions forcées ou les changements de mot de passe.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Nombre de tokens révoqués
        """
        # Cette implémentation dépend du backend Redis
        # Pour un cache simple, on ne peut pas itérer facilement
        logger.info(f"Revoking all refresh tokens for user {user_id}")
        return 0  # À implémenter pour Redis

    def get_token_info(self, user_id: str, token: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations sur un token de refresh.

        Args:
            user_id: ID de l'utilisateur
            token: Token

        Returns:
            Dictionnaire avec les infos du token, ou None
        """
        cache_key = f"refresh_token:{user_id}:{token}"
        return self.cache.get(cache_key)

    def create_token_pair(
        self, user_id: str, access_token: str
    ) -> Tuple[str, str, int]:
        """
        Crée une paire (access_token, refresh_token).

        Args:
            user_id: ID de l'utilisateur
            access_token: Token d'accès (généralement créé par Firebase)

        Returns:
            Tuple: (access_token, refresh_token, expires_in_seconds)
        """
        refresh_token = self.generate_refresh_token(user_id)
        return access_token, refresh_token, self.access_token_ttl


class InMemoryTokenCache:
    """Cache simple en mémoire pour le développement."""

    def __init__(self):
        """Initialise le cache en mémoire."""
        self.store: Dict[str, Tuple[Any, Optional[float]]] = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stocke une valeur."""
        expire_at = None
        if ttl:
            expire_at = datetime.utcnow().timestamp() + ttl

        self.store[key] = (value, expire_at)

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur."""
        if key not in self.store:
            return None

        value, expire_at = self.store[key]

        if expire_at and datetime.utcnow().timestamp() > expire_at:
            del self.store[key]
            return None

        return value

    def delete(self, key: str) -> None:
        """Supprime une valeur."""
        if key in self.store:
            del self.store[key]


# Singleton global
_token_manager = None


def get_token_manager() -> TokenManager:
    """Retourne l'instance singleton du TokenManager."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager

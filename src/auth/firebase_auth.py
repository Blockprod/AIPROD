"""
Authentification et vérification JWT avec Firebase.
Implémente la validation des tokens Bearer.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FirebaseAuthenticator:
    """Validateur de tokens Firebase JWT."""

    def __init__(self):
        self.enabled = os.getenv("FIREBASE_ENABLED", "true").lower() == "true"
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.firebase_app = None

        if self.enabled:
            self._init_firebase()

    def _init_firebase(self):
        """Initialise le client Firebase."""
        try:
            import firebase_admin
            from firebase_admin import credentials, auth

            # En développement, utiliser env var
            # En production (Cloud Run), utiliser Application Default Credentials
            if not firebase_admin._apps:
                if os.getenv("FIREBASE_CREDENTIALS_PATH"):
                    creds = credentials.Certificate(
                        os.getenv("FIREBASE_CREDENTIALS_PATH")
                    )
                    firebase_admin.initialize_app(creds, {"projectId": self.project_id})
                else:
                    # Application Default Credentials (Cloud Run)
                    firebase_admin.initialize_app()

            self.firebase_app = firebase_admin
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.enabled = False

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Vérifie un token JWT Firebase.

        Args:
            token: Token Bearer (sans le préfixe "Bearer ")

        Returns:
            Dictionnaire avec claims du token, ou None si invalide
        """
        if not self.enabled:
            logger.warning("Firebase authentication is disabled")
            return None

        try:
            from firebase_admin import auth

            decoded = auth.verify_id_token(token)
            return decoded
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            return None

    def verify_custom_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Vérifie un token JWT personnalisé généré par un backend.
        """
        try:
            from firebase_admin import auth

            decoded = auth.verify_session_cookie(token, check_revoked=True)
            return decoded
        except Exception as e:
            logger.warning(f"Custom token verification failed: {e}")
            return None

    def get_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Extrait les infos utilisateur du token.
        """
        decoded = self.verify_token(token)
        if not decoded:
            return None

        return {
            "uid": decoded.get("uid"),
            "email": decoded.get("email"),
            "email_verified": decoded.get("email_verified"),
            "custom_claims": decoded.get("custom_claims", {}),
            "iat": datetime.fromtimestamp(decoded.get("iat", 0)),
            "exp": datetime.fromtimestamp(decoded.get("exp", 0)),
        }


# Singleton global
_authenticator = None


def get_firebase_authenticator() -> FirebaseAuthenticator:
    """Retourne l'instance singleton du Firebase Authenticator."""
    global _authenticator
    if _authenticator is None:
        _authenticator = FirebaseAuthenticator()
    return _authenticator


def extract_token_from_header(authorization: str) -> Optional[str]:
    """
    Extrait le token du header Authorization.

    Format attendu: "Bearer <token>"
    """
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning(f"Invalid authorization header format")
        return None

    return parts[1]

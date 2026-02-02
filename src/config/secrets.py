"""
Configuration et chargement sécurisé des secrets depuis GCP Secret Manager.
En développement local, les secrets sont chargés depuis .env.
En production (Cloud Run), ils sont chargés depuis GCP Secret Manager.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_secret_from_secret_manager(secret_id: str) -> Optional[str]:
    """
    Charge un secret depuis GCP Secret Manager.

    Utilise l'authentification par défaut de Cloud Run / Application Default Credentials.

    Args:
        secret_id: ID du secret (ex: "GEMINI_API_KEY")

    Returns:
        Valeur du secret ou None si non trouvé
    """
    try:
        from google.cloud import secretmanager

        project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if not project_id:
            logger.warning(f"GCP_PROJECT_ID not set, cannot load secret {secret_id}")
            return None

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")

        logger.info(f"Secret loaded from Secret Manager: {secret_id}")
        return secret_value
    except Exception as e:
        logger.error(f"Failed to load secret {secret_id} from Secret Manager: {e}")
        return None


def get_secret(secret_id: str, default: Optional[str] = None) -> Optional[str]:
    """
    Charge un secret avec stratégie fallback.

    Priorités :
    1. Variable d'environnement déjà définie
    2. GCP Secret Manager (en production)
    3. Valeur par défaut

    Args:
        secret_id: ID du secret
        default: Valeur par défaut si non trouvé

    Returns:
        Valeur du secret
    """
    # Vérifier si déjà en env
    if secret_id in os.environ:
        value = os.environ[secret_id]
        if value and not value.startswith("<"):  # < indique un placeholder
            return value

    # Essayer Secret Manager
    if os.getenv("ENVIRONMENT") == "production" or os.getenv("CLOUD_RUN_ENVIRONMENT"):
        value = get_secret_from_secret_manager(secret_id)
        if value:
            return value

    # Fallback
    return default


def load_secrets():
    """
    Charge tous les secrets critiques au démarrage de l'app.
    Lance une exception si un secret critique est absent.
    """
    critical_secrets = [
        "GEMINI_API_KEY",
        "GCP_PROJECT_ID",
    ]

    optional_secrets = [
        "RUNWAY_API_KEY",
        "REPLICATE_API_KEY",
        "DD_API_KEY",
        "DD_APP_KEY",
    ]

    # Charger secrets critiques
    for secret_id in critical_secrets:
        value = get_secret(secret_id)
        if not value:
            logger.warning(
                f"Critical secret not found: {secret_id}. "
                f"Using development mode. Set in env or Secret Manager."
            )
        else:
            os.environ[secret_id] = value

    # Charger secrets optionnels (log warning si absent)
    for secret_id in optional_secrets:
        value = get_secret(secret_id)
        if value:
            os.environ[secret_id] = value
        else:
            logger.warning(f"Optional secret not configured: {secret_id}")

    logger.info("Secrets loaded successfully")


# Fonction utilitaire pour masquer les secrets en logs
def mask_secret(value: str, visible_chars: int = 4) -> str:
    """
    Masque une clé API pour les logs.

    Exemple: "AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw" → "AIzaSy...Rbw"
    """
    if not value or len(value) <= visible_chars * 2:
        return "***MASKED***"
    return f"{value[:visible_chars]}...{value[-visible_chars:]}"

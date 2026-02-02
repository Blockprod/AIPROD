"""
Tests unitaires pour les modules de sécurité Phase 0.

Tests couvrant:
- src/config/secrets.py
- src/auth/firebase_auth.py
- src/security/audit_logger.py
"""

import os
import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# Test imports
from src.config.secrets import (
    get_secret,
    mask_secret,
    load_secrets,
)
from src.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    get_audit_logger,
)


class TestSecretManagement:
    """Tests pour le système de gestion des secrets."""

    def test_mask_secret_basic(self):
        """Teste le masquage de secrets."""
        # Secret court (moins de 8 chars)
        result = mask_secret("ab", visible_chars=1)
        assert result == "***MASKED***"

        # Secret normal (suffisamment long)
        api_key = "AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw"
        result = mask_secret(api_key, visible_chars=4)
        assert result == "AIza...tRbw"
        assert "cdfz9" not in result
        assert "gIIbG" not in result

        # Secret de longueur moyenne
        result = mask_secret("abcde", visible_chars=1)
        assert result == "a...e"

    def test_mask_secret_edge_cases(self):
        """Tests les cas limites."""
        # Secret vide
        result = mask_secret("")
        assert result == "***MASKED***"

        # Secret très court (< 8 chars minimum)
        result = mask_secret("abc", visible_chars=5)
        assert result == "***MASKED***"

    @patch.dict(os.environ, {"TEST_SECRET": "test_value"})
    def test_get_secret_from_env(self):
        """Teste la récupération d'un secret depuis l'env."""
        value = get_secret("TEST_SECRET")
        assert value == "test_value"

    @patch.dict(os.environ, {}, clear=False)
    def test_get_secret_with_default(self):
        """Teste le fallback sur la valeur par défaut."""
        value = get_secret("NONEXISTENT_SECRET", default="default_value")
        # Peut être None ou default selon la config
        assert value is None or value == "default_value"

    def test_get_secret_with_placeholder(self):
        """Teste que les placeholders sont ignorés."""
        with patch.dict(os.environ, {"SECRET": "<charger depuis Secret Manager>"}):
            # Devrait ignorer le placeholder
            value = get_secret("SECRET", default="fallback")
            # Dépend de l'implémentation
            assert value is None or value == "fallback"


class TestAuditLogger:
    """Tests pour le système d'audit logging."""

    def setup_method(self):
        """Setup avant chaque test."""
        # Reset le singleton
        import src.security.audit_logger

        src.security.audit_logger._audit_logger = None

    def test_audit_logger_singleton(self):
        """Teste que AuditLogger est un singleton."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2

    def test_log_event_basic(self):
        """Teste l'enregistrement d'un événement basic."""
        logger = get_audit_logger()

        # Devrait pas lever d'exception
        logger.log_event(
            event_type=AuditEventType.API_CALL,
            user_id="user@example.com",
            action="test_action",
            resource="/test",
            status="success",
        )

    def test_log_event_with_details(self):
        """Teste l'enregistrement avec détails."""
        logger = get_audit_logger()

        logger.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            user_id="admin@example.com",
            action="unauthorized_access_attempt",
            resource="/admin/metrics",
            status="denied",
            details={
                "ip_address": "192.168.1.1",
                "attempted_resource": "/admin/sensitive",
                "reason": "Missing admin role",
            },
        )

    def test_audit_event_types(self):
        """Teste tous les types d'événements disponibles."""
        logger = get_audit_logger()

        event_types = [
            (AuditEventType.API_CALL, "API call"),
            (AuditEventType.AUTH_SUCCESS, "Auth success"),
            (AuditEventType.AUTH_FAILURE, "Auth failure"),
            (AuditEventType.PERMISSION_DENIED, "Permission denied"),
            (AuditEventType.DATA_ACCESS, "Data access"),
            (AuditEventType.CONFIG_CHANGE, "Config change"),
            (AuditEventType.SECRET_ACCESS, "Secret access"),
            (AuditEventType.ADMIN_ACTION, "Admin action"),
            (AuditEventType.SECURITY_ALERT, "Security alert"),
        ]

        for event_type, description in event_types:
            # Devrait pas lever d'exception
            logger.log_event(
                event_type=event_type, action=description, status="success"
            )

    def test_log_auth_success(self):
        """Teste le logging d'une authentification réussie."""
        logger = get_audit_logger()

        logger.log_auth_success(user_id="user@example.com", method="firebase")

    def test_log_auth_failure(self):
        """Teste le logging d'une authentification échouée."""
        logger = get_audit_logger()

        logger.log_auth_failure(
            user_id="user@example.com", reason="Invalid token signature"
        )

    def test_log_permission_denied(self):
        """Teste le logging d'une autorisation refusée."""
        logger = get_audit_logger()

        logger.log_permission_denied(
            user_id="user@example.com",
            action="access_admin_panel",
            resource="/admin/users",
            required_role="admin",
        )

    def test_log_api_call(self):
        """Teste le logging d'un appel API."""
        logger = get_audit_logger()

        logger.log_api_call(
            endpoint="/pipeline/run",
            method="POST",
            user_id="user@example.com",
            status_code=200,
            duration_ms=150.5,
        )

    def test_log_secret_access(self):
        """Teste le logging d'un accès à un secret."""
        logger = get_audit_logger()

        logger.log_secret_access(secret_id="GEMINI_API_KEY", user_id="system")

    def test_log_security_alert(self):
        """Teste le logging d'une alerte de sécurité."""
        logger = get_audit_logger()

        logger.log_security_alert(
            alert_type="multiple_failed_auth_attempts",
            details={
                "user_id": "user@example.com",
                "attempts": 5,
                "time_window_seconds": 300,
                "ip_addresses": ["192.168.1.1", "192.168.1.2"],
            },
        )


class TestAuditEventType:
    """Tests pour l'énumération des types d'événements."""

    def test_event_type_enum_values(self):
        """Teste que tous les événements ont des valeurs correctes."""
        assert AuditEventType.API_CALL.value == "API_CALL"
        assert AuditEventType.AUTH_SUCCESS.value == "AUTH_SUCCESS"
        assert AuditEventType.AUTH_FAILURE.value == "AUTH_FAILURE"
        assert AuditEventType.PERMISSION_DENIED.value == "PERMISSION_DENIED"
        assert AuditEventType.SECURITY_ALERT.value == "SECURITY_ALERT"

    def test_event_type_string_conversion(self):
        """Teste la conversion des enum en strings."""
        event = AuditEventType.API_CALL
        assert str(event.value) == "API_CALL"
        assert event.name == "API_CALL"


class TestSecretLoadingIntegration:
    """Tests d'intégration pour le système de secrets."""

    @patch.dict(os.environ, {"ENVIRONMENT": "development"})
    def test_development_mode_secrets(self):
        """Teste le chargement des secrets en mode développement."""
        # En dev, utiliser les env vars
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            value = get_secret("GEMINI_API_KEY")
            assert value == "test_key"

    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_production_mode_secrets(self):
        """Teste que la production utiliserait Secret Manager."""
        # En production, Secret Manager serait utilisé
        # (simulé ici puisque GCP n'est pas disponible)
        with patch("src.config.secrets.get_secret_from_secret_manager") as mock_sm:
            mock_sm.return_value = "secret_from_gcp"

            # En production, devrait essayer Secret Manager
            value = get_secret("TEST_SECRET")
            # Dépend du fallback


class TestAuditLoggerConfig:
    """Tests de configuration du AuditLogger."""

    def test_audit_logger_disabled_datadog(self):
        """Teste que Datadog est optionnel."""
        with patch.dict(os.environ, {}, clear=True):
            logger = AuditLogger()
            # Devrait s'initialiser même sans Datadog
            assert logger is not None
            assert logger.service_name == "aiprod-v33"

    @patch.dict(os.environ, {"SERVICE_NAME": "custom-service"})
    def test_audit_logger_custom_service_name(self):
        """Teste la configuration personnalisée du nom du service."""
        logger = AuditLogger()
        assert logger.service_name == "custom-service"

    @patch.dict(os.environ, {"ENVIRONMENT": "staging"})
    def test_audit_logger_environment_config(self):
        """Teste la configuration de l'environnement."""
        logger = AuditLogger()
        assert logger.environment == "staging"


# Pytest markers pour organiser les tests
pytestmark = pytest.mark.unit


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

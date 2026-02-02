#!/usr/bin/env python
"""
Test simple pour vérifier que l'audit logger fonctionne
"""

from src.security.audit_logger import get_audit_logger, AuditEventType


def test_audit_logger():
    """Test que l'audit logger peut être créé et utilisé."""

    try:
        # Créer un logger d'audit
        audit_logger = get_audit_logger()

        print("✅ Audit logger créé avec succès")

        # Vérifier les méthodes
        assert hasattr(audit_logger, "log_api_call"), "log_api_call method missing"
        assert hasattr(audit_logger, "log_event"), "log_event method missing"
        print("✅ Toutes les méthodes requises existent")

        # Test log_api_call
        audit_logger.log_api_call(
            endpoint="/pipeline/run",
            method="POST",
            user_id="test@example.com",
            status_code=200,
            duration_ms=1234.5,
        )
        print("✅ log_api_call() exécuté avec succès")

        # Test log_event
        audit_logger.log_event(
            event_type=AuditEventType.ADMIN_ACTION,
            user_id="admin@example.com",
            action="test_action",
            details={"test": "data"},
        )
        print("✅ log_event() exécuté avec succès")

        # Test autre endpoint
        audit_logger.log_api_call(
            endpoint="/metrics",
            method="GET",
            user_id="user@example.com",
            status_code=200,
            duration_ms=45.2,
        )
        print("✅ Deuxième log_api_call() exécuté avec succès")

        print("\n✅ ÉTAPE 5: Audit Logger functional tests - ALL PASSED!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = test_audit_logger()
    sys.exit(0 if success else 1)

"""
GUIDE D'INTÉGRATION: Phase 0 Security Middleware

Ce fichier montre les modifications à apporter à src/api/main.py
pour intégrer l'authentification, l'autorisation et l'audit logging.

ÉTAPES:

1. Ajouter les imports au début de main.py
2. Charger les secrets au démarrage
3. Ajouter le middleware d'authentification
4. Protéger les endpoints critiques
5. Ajouter les logs d'audit
   """

# ==============================================================================

# ÉTAPE 1: Ajouter les imports en haut de main.py

# ==============================================================================

# Ajouter après les imports existants:

from src.config.secrets import load_secrets, get_secret, mask_secret
from src.auth.firebase_auth import get_firebase_authenticator
from src.api.auth_middleware import (
verify_token,
optional_verify_token,
AuthMiddleware,
require_auth
)
from src.security.audit_logger import (
get_audit_logger,
AuditEventType,
audit_log
)

# ==============================================================================

# ÉTAPE 2: Charger les secrets au démarrage (après create_app)

# ==============================================================================

# Ajouter au démarrage de main.py, AVANT de définir les routes:

@app.on_event("startup")
async def startup_event():
"""Initialise les secrets et la configuration au démarrage."""
logger.info("🔐 Initializing security components...")

    # Charger les secrets depuis GCP Secret Manager / .env
    load_secrets()
    logger.info("✅ Secrets loaded successfully")

    # Initialiser Firebase Authentication
    auth = get_firebase_authenticator()
    if auth.enabled:
        logger.info("✅ Firebase Authentication initialized")
    else:
        logger.warning("⚠️  Firebase Authentication disabled (development mode)")

    # Initialiser Audit Logger
    audit_logger = get_audit_logger()
    logger.info("✅ Audit logging initialized")

    logger.info("🔐 Security initialization complete")

@app.on_event("shutdown")
async def shutdown_event():
"""Nettoie les ressources au arrêt."""
logger.info("🛑 Shutting down...") # Optionnel: fermer les connexions Datadog, etc.

# ==============================================================================

# ÉTAPE 3: Ajouter le middleware d'authentification

# ==============================================================================

# Ajouter après Instrumentator().instrument(app):

app.add_middleware(AuthMiddleware)

# ==============================================================================

# ÉTAPE 4: ROUTES PUBLIQUES (sans authentification requise)

# ==============================================================================

# Ces routes restent inchangées (publiques):

@app.get("/")
async def root() -> Dict[str, str]:
"""Endpoint public - accueil."""
logger.info("GET /")
return {
"status": "ok",
"name": "AIPROD V33 API",
"docs": "/docs",
"openapi": "/openapi.json",
}

@app.get("/health")
async def health() -> Dict[str, str]:
"""Endpoint public - santé de l'API."""
logger.info("GET /health")
return {"status": "ok"}

# ==============================================================================

# ÉTAPE 5: ROUTES PROTÉGÉES (authentification obligatoire)

# ==============================================================================

# MODIFICATION 1: Ajouter authentification et audit à /pipeline/run

@app.post("/pipeline/run")
@audit_log(AuditEventType.API_CALL, action="pipeline_start")
async def run_pipeline(
request: PipelineRequest,
user: dict = Depends(verify_token) # <-- AJOUTER CETTE LIGNE
) -> PipelineResponse:
"""
Lance l'exécution du pipeline complet.

    🔐 AUTHENTIFICATION REQUISE

    Supporte les presets: quick_social, brand_campaign, premium_spot

    Args:
        request (PipelineRequest): Requête avec paramètres du pipeline.
        user: Utilisateur authentifié (injecté par verify_token)
    Returns:
        PipelineResponse: Résultat du pipeline.
    """
    try:
        logger.info(f"POST /pipeline/run from {user['email']} with content={request.content[:50]}, preset={request.preset}")
        start_time = time.time()

        # Ajouter l'ID utilisateur aux métadonnées
        request_data = request.model_dump()
        request_data["_user_id"] = user.get("uid")
        request_data["_user_email"] = user.get("email")

        # [... reste du code inchangé ...]

        # Ajouter audit log de succès
        audit_logger = get_audit_logger()
        latency_ms = (time.time() - start_time) * 1000
        audit_logger.log_api_call(
            endpoint="/pipeline/run",
            method="POST",
            user_id=user.get("email"),
            status_code=200,
            duration_ms=latency_ms
        )

        return PipelineResponse(
            status="success",
            state=state_machine.state.name,
            data=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")

        # Log l'erreur
        audit_logger = get_audit_logger()
        audit_logger.log_api_call(
            endpoint="/pipeline/run",
            method="POST",
            user_id=user.get("email"),
            status_code=500,
            duration_ms=(time.time() - start_time) * 1000
        )

        metrics_collector.record_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================

# ÉTAPE 6: ROUTES ADMIN (authentification + vérification de role)

# ==============================================================================

# MODIFICATION 2: Protéger les endpoints d'administration

@app.get("/admin/metrics")
@require_auth(required_roles=["admin"])
async def get_admin_metrics(
user: dict = Depends(verify_token)
) -> Dict[str, Any]:
"""
Dashboard admin avec métriques détaillées.

    🔐 AUTHENTIFICATION REQUISE + ROLE 'admin'
    """
    audit_logger = get_audit_logger()
    audit_logger.log_api_call(
        endpoint="/admin/metrics",
        method="GET",
        user_id=user.get("email"),
        status_code=200
    )

    return {
        "total_jobs": len(job_manager.jobs),
        "metrics": metrics_collector.get_summary(),
    }

@app.post("/admin/reset")
@require_auth(required_roles=["admin"])
async def admin_reset(user: dict = Depends(verify_token)) -> Dict[str, str]:
"""
Endpoint administrateur pour réinitialiser l'état.

    🔐 AUTHENTIFICATION REQUISE + ROLE 'admin'
    """
    audit_logger = get_audit_logger()

    # Log l'action sensible
    audit_logger.log_event(
        event_type=AuditEventType.ADMIN_ACTION,
        user_id=user.get("email"),
        action="admin_reset",
        details={"endpoint": "/admin/reset"}
    )

    # Exécuter l'action
    job_manager.clear_all()

    return {"status": "reset_complete"}

# ==============================================================================

# ÉTAPE 7: ROUTES SEMI-PUBLIQUES (authentification optionnelle)

# ==============================================================================

# MODIFICATION 3: Ajouter authentification optionnelle

@app.get("/jobs/{job_id}")
async def get_job(
job_id: str,
user: Optional[dict] = Depends(optional_verify_token)
) -> Dict[str, Any]:
"""
Récupère les infos d'un job.

    🔓 PUBLIQUE (mais enregistre l'utilisateur si authentifié)
    """
    audit_logger = get_audit_logger()

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Log l'accès
    audit_logger.log_api_call(
        endpoint=f"/jobs/{job_id}",
        method="GET",
        user_id=user.get("email") if user else "anonymous",
        status_code=200
    )

    return job

# ==============================================================================

# ÉTAPE 8: LOGGING DES ERREURS D'AUTHENTIFICATION

# ==============================================================================

# Ajouter un exception handler personnalisé:

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
"""Gère les exceptions HTTP avec audit logging."""
audit_logger = get_audit_logger()

    # Log l'erreur d'authentification
    if exc.status_code == 401:
        audit_logger.log_auth_failure(
            user_id=None,
            reason=exc.detail
        )
    elif exc.status_code == 403:
        audit_logger.log_permission_denied(
            user_id=request.get("user", {}).get("email", "unknown"),
            action=f"{request.method} {request.url.path}",
            resource=request.url.path
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# ==============================================================================

# RÉSUMÉ DES CHANGEMENTS

# ==============================================================================

"""
FICHIER MODIFIÉ: src/api/main.py

CHANGEMENTS À APPORTER:

1. ✅ Imports (ajouter 15 lignes)
   - src.config.secrets
   - src.auth.firebase_auth
   - src.api.auth_middleware
   - src.security.audit_logger

2. ✅ Startup (ajouter 20 lignes)
   - @app.on_event("startup") pour charger secrets et initialiser auth

3. ✅ Middleware (ajouter 1 ligne)
   - app.add_middleware(AuthMiddleware)

4. ✅ Route /pipeline/run (modifier)
   - Ajouter user: dict = Depends(verify_token)
   - Ajouter audit logs
   - Ajouter user_id aux métadonnées

5. ✅ Nouvelles routes admin (ajouter)
   - /admin/metrics (protégé, role=admin)
   - /admin/reset (protégé, role=admin)

6. ✅ Routes existantes (modifier optionnel)
   - /pipeline/status → ajouter authentification
   - /metrics → renommer en /internal/metrics pour éviter conflit Prometheus
   - /alerts → ajouter authentification

7. ✅ Exception handler (ajouter)
   - Pour logger les erreurs d'authentification

TOTAL: ~100-150 lignes de code à ajouter/modifier

IMPACT SUR LES PERFORMANCES:

- +5-10ms par requête (vérification JWT)
- Caching possible pour tokens (à implémenter en P1)

TESTS À EFFECTUER:

- curl -X POST http://localhost:8000/pipeline/run -d '...' (sans token → 401)
- curl -X POST -H "Authorization: Bearer <token>" http://localhost:8000/pipeline/run -d '...' (avec token → 200)
- Vérifier logs d'audit dans Cloud Logging / stdout
  """

# ==============================================================================

# FICHIERS ASSOCIÉS

# ==============================================================================

"""
Fichiers créés (P0):

- src/config/secrets.py ✅ Loader GCP Secret Manager
- src/auth/firebase_auth.py ✅ Vérification JWT Firebase
- src/api/auth_middleware.py ✅ Dépendances FastAPI
- src/security/audit_logger.py ✅ Logging d'audit
- requirements.txt ✅ Mises à jour dépendances
- .env.example ✅ Template sécurisé

Fichiers à modifier (P0):

- src/api/main.py 🔄 À faire dans prochaine étape

Fichiers à créer manuellement:

- credentials/firebase-adminsdk.json (télécharger depuis Firebase Console)
- .env.local (pour développement, créer à partir de .env.example)
  """

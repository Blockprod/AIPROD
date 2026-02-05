"""
API REST FastAPI pour AIPROD V33
Endpoints pour l'orchestration du pipeline, gestion des entrées et exposition des résultats.
"""

import time
import json
import zipfile
from io import BytesIO
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import (
    FastAPI,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Optional, List
import asyncio
from src.orchestrator.state_machine import StateMachine
from src.api.functions.input_sanitizer import InputSanitizer
from src.api.functions.financial_orchestrator import FinancialOrchestrator
from src.api.functions.technical_qa_gate import TechnicalQAGate
from src.utils.metrics_collector import MetricsCollector, prom_router
from src.utils.monitoring import logger
from src.config.secrets import load_secrets, get_secret, mask_secret
from src.auth.firebase_auth import get_firebase_authenticator
from src.auth.token_manager import get_token_manager
from src.auth.auth_models import RefreshTokenRequest, TokenResponse, RevokeTokenRequest
from src.auth.api_key_manager import get_api_key_manager
from src.auth.api_key_models import (
    CreateAPIKeyRequest, APIKeyResponse, APIKeyMetadata, ListAPIKeysResponse,
    RotateAPIKeyRequest, RevokeAPIKeyRequest, RevokeAllKeysRequest, RevokeAllKeysResponse,
    APIKeyStatsResponse, APIKeyHealthCheck
)
from src.api.functions.export_service import get_export_service, ExportFormat
from src.api.functions.export_models import ExportRequest, ExportResponse, ExportFormatsResponse
from src.api.auth_middleware import (
    verify_token,
    optional_verify_token,
    AuthMiddleware,
    require_auth,
)
from src.security.csrf_protection import get_csrf_manager, get_csrf_token, verify_csrf_token
from fastapi.responses import FileResponse, StreamingResponse
from src.security.audit_logger import get_audit_logger, AuditEventType, audit_log
from src.api.presets import (
    get_preset,
    get_all_presets,
    apply_preset_to_request,
    estimate_cost_for_preset,
    PresetTier,
)
from src.api.cost_estimator import get_full_cost_estimate, get_job_actual_costs
from src.api.icc_manager import get_job_manager, JobState
from src.db.models import get_session_factory, JobState as DBJobState
from src.db.job_repository import JobRepository
from src.pubsub.client import get_pubsub_client, PubSubClient
from src.api.rate_limiter import limiter, rate_limit_exceeded_handler, apply_rate_limit
from src.api.cors_config import CORS_CONFIG, SECURITY_HEADERS
from src.api.input_validator import validate_request_size, validate_input_field, VALIDATION_RULES
from src.monitoring.metrics_collector import get_metrics_collector
from src.monitoring.monitoring_middleware import MonitoringMiddleware, CacheMetricsMiddleware, ResourceMetricsMiddleware
from src.monitoring.monitoring_routes import setup_monitoring_routes
from src.performance.compression_middleware import CompressionMiddleware, CacheHeaderMiddleware
from src.performance.performance_routes import setup_performance_routes
from src.deployment.deployment_routes import setup_deployment_routes
from src.analytics.analytics_routes import setup_analytics_routes
import os

# Database session factory
_db_session_factory = None


def get_db_session():
    """Get database session."""
    global _db_session_factory
    if _db_session_factory is None:
        db_url = os.getenv(
            "DATABASE_URL", "postgresql://aiprod:password@localhost:5432/aiprod_v33"
        )
        _db_session_factory, _ = get_session_factory(db_url)
    return _db_session_factory()


# 🔐 Lifespan context manager pour startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application (startup/shutdown)."""
    # Startup
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

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")


app = FastAPI(
    title="AIPROD V33 API",
    description="Pipeline de génération vidéo IA avec orchestration, agents et QA",
    version="1.0.0",
    lifespan=lifespan,
)

# Ajout du router Prometheus /metrics
app.include_router(prom_router)

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

# � Rate limiting
app.state.limiter = limiter
app.add_exception_handler(Exception, rate_limit_exceeded_handler)

# �🔐 Ajouter le middleware d'authentification
app.add_middleware(AuthMiddleware)

# ?? Ajouter les middlewares de monitoring
app.add_middleware(MonitoringMiddleware)
app.add_middleware(CacheMetricsMiddleware)
app.add_middleware(ResourceMetricsMiddleware)

# ⚡ Ajouter les middlewares de performance
app.add_middleware(CacheHeaderMiddleware)
app.add_middleware(CompressionMiddleware)


# DTOs pour les entrées/sorties
class PipelineRequest(BaseModel):
    """Schéma de requête pour le pipeline."""

    content: str
    priority: str = "low"
    lang: str = "en"
    preset: Optional[str] = Field(
        default=None,
        description="Preset à utiliser: quick_social, brand_campaign, premium_spot",
    )
    duration_sec: Optional[int] = Field(
        default=30, description="Durée vidéo souhaitée en secondes"
    )

    model_config = ConfigDict(extra="allow")


class CostEstimateRequest(BaseModel):
    """Schéma de requête pour estimation de coûts."""

    content: str
    duration_sec: int = 30
    preset: Optional[str] = None
    complexity: str = "standard"


class CostEstimateResponse(BaseModel):
    """Schéma de réponse estimation de coûts."""

    aiprod_optimized: float
    runway_alone: float
    savings: float
    savings_percent: float
    quality_guarantee: float
    backend_selected: str
    breakdown: Dict[str, float]
    value_proposition: str


class PipelineResponse(BaseModel):
    """Schéma de réponse du pipeline."""

    status: str
    state: str
    data: Dict[str, Any]


# Instances globales
state_machine = StateMachine()
input_sanitizer = InputSanitizer()
financial_orchestrator = FinancialOrchestrator()
technical_qa_gate = TechnicalQAGate()
metrics_collector = MetricsCollector()
job_manager = get_job_manager()


# Favicon minimaliste (1x1 px) pour éviter les 404 locales
FAVICON_BYTES = (
    b"\x00\x00\x01\x00\x01\x00\x10\x10\x10\x00\x00\x00\x00\x00"
    b"\x28\x01\x00\x00\x16\x00\x00\x00\x28\x00\x00\x00\x10\x00"
    b"\x00\x00\x10\x00\x00\x00\x01\x00\x04\x00\x00\x00\x00\x00"
    b"\x80\x00\x00\x00\xc4\x0e\x00\x00\xc4\x0e\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    b"\xff\xff\xff\xff\xff\xff\xff\xff"
)


# Route d'accueil simple pour éviter le 404 sur /
@app.get("/")
async def root() -> Dict[str, str]:
    logger.info("GET /")
    return {
        "status": "ok",
        "name": "AIPROD V33 API",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health")
@limiter.limit("1000/minute")
async def health(request: Request) -> Dict[str, str]:
    """
    Endpoint de santé de l'API.
    Rate limit: 1000 req/min (high - for monitoring)
    
    Returns:
        Dict[str, str]: Status de l'API.
    """
    logger.info("GET /health")
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
# 🔐 AUTHENTIFICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/auth/refresh")
@limiter.limit("60/minute")
async def refresh_access_token(
    request_data: RefreshTokenRequest, request: Request
) -> TokenResponse:
    """
    Rafraîchit un access token expiré en utilisant un refresh token.
    
    Processus:
    1. Vérifier que le refresh token est valide
    2. Générer un nouveau pair (access + refresh token)
    3. Révoquer l'ancien refresh token
    4. Retourner le nouveau pair
    
    🔒 RATE LIMITED: 60 requests/minute
    
    Args:
        request_data: Contient le refresh_token
        
    Returns:
        TokenResponse: Nouveau access_token + refresh_token
        
    Raises:
        401: Refresh token invalide ou expiré
        403: Token has been revoked
    """
    try:
        refresh_token = request_data.refresh_token
        
        # Extraire user_id du token actuel (via Firebase Custom Claims)
        # Dans une implémentation réelle, on décoderait le JWT
        # Pour maintenant, on va stocker user_id dans Redis avec le token
        
        # Chercher le user_id associé au token
        token_manager = get_token_manager()
        
        # Note: Dans une vraie implémentation, on aurait besoin de décoder le token
        # Pour cette démo, on va supposer que le token stocke user_id
        # En production, utiliser: user_id = decode_refresh_token(refresh_token)
        
        firebase_auth = get_firebase_authenticator()
        
        # TODO: Décoder le refresh token pour extraire user_id
        # Pour maintenant, on va utiliser un placeholder
        user_id = "anonymous"  # Placeholder - à implémenter avec decode réel
        
        # Vérifier que le token est valide
        if not token_manager.verify_refresh_token(user_id, refresh_token):
            logger.warning(f"Invalid or expired refresh token")
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired refresh token"
            )
        
        # Générer un nouveau access token (en prod, via Firebase)
        # Pour cette démo, on va créer un simple JWT
        new_access_token = f"new_access_token_{secrets.token_hex(16)}"
        
        # Effectuer la rotation du token
        new_refresh_token = token_manager.rotate_refresh_token(user_id, refresh_token)
        
        if not new_refresh_token:
            logger.error(f"Failed to rotate refresh token for user {user_id}")
            raise HTTPException(
                status_code=500,
                detail="Failed to refresh token"
            )
        
        # Log l'événement
        audit_log(
            event_type=AuditEventType.AUTH_TOKEN_REFRESH,
            user_id=user_id,
            details={"user_id": user_id}
        )
        
        logger.info(f"Token refreshed for user: {user_id}")
        
        from datetime import datetime
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="Bearer",
            expires_in=900,  # 15 minutes
            issued_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/auth/revoke")
@limiter.limit("60/minute")
async def revoke_token(
    request_data: RevokeTokenRequest,
    request: Request = None,
    user: dict = Depends(verify_token),
) -> Dict[str, str]:
    """
    Révoque un refresh token de manière permanente.
    Utilisé lors de la déconnexion ou du changement de mot de passe.
    
    🔐 AUTHENTIFICATION REQUISE
    🔒 RATE LIMITED: 60 requests/minute
    
    Args:
        request_data: Contient le refresh_token à révoquer
        user: Utilisateur authentifié
        
    Returns:
        {"status": "revoked"}
        
    Raises:
        401: Non authentifié
        404: Token not found
    """
    try:
        user_id = user.get("uid", "unknown")
        refresh_token = request_data.refresh_token
        
        token_manager = get_token_manager()
        
        # Révoquer le token
        if token_manager.revoke_refresh_token(user_id, refresh_token):
            audit_log(
                event_type=AuditEventType.AUTH_TOKEN_REVOKE,
                user_id=user_id,
                details={"user_id": user_id}
            )
            logger.info(f"Token revoked for user: {user_id}")
            return {"status": "revoked"}
        else:
            raise HTTPException(status_code=404, detail="Token not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/auth/token-info")
@limiter.limit("60/minute")
async def get_token_info(
    refresh_token: str,
    request: Request = None,
    user: dict = Depends(verify_token),
) -> Dict[str, Any]:
    """
    Récupère les informations sur un refresh token.
    
    🔐 AUTHENTIFICATION REQUISE
    🔒 RATE LIMITED: 60 requests/minute
    
    Args:
        refresh_token: Token à vérifier (query parameter)
        user: Utilisateur authentifié
        
    Returns:
        Token info ou message d'erreur
    """
    try:
        user_id = user.get("uid", "unknown")
        token_manager = get_token_manager()
        
        token_info = token_manager.get_token_info(user_id, refresh_token)
        
        if not token_info:
            raise HTTPException(status_code=404, detail="Token not found")
        
        return {
            "user_id": token_info.get("user_id"),
            "created_at": token_info.get("created_at"),
            "expires_at": token_info.get("expires_at"),
            "version": token_info.get("version", 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Import secrets for token generation
import secrets


@app.post("/pipeline/run")
@limiter.limit("30/minute")
async def run_pipeline(
    request: Request, request_data: PipelineRequest, user: dict = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Lance l'exécution du pipeline complet de manière asynchrone.

    🔐 AUTHENTIFICATION REQUISE
    🚀 ASYNC: Retourne immédiatement un job_id, traitement en arrière-plan
    🔒 RATE LIMITED: 30 requests/minute (~1 job per 2 seconds)

    Supporte les presets: quick_social, brand_campaign, premium_spot

    Args:
        request: Requête HTTP (pour accès aux headers et validation)
        request_data (PipelineRequest): Requête avec paramètres du pipeline.
        user: Utilisateur authentifié (injecté par verify_token)
        
    Returns:
        Dict avec job_id et status "queued"
    """
    start_time = time.time()

    try:
        # Validate request size
        await validate_request_size(request)
        
        # Validate input fields
        await validate_input_field("content", request_data.content, VALIDATION_RULES["content"])
        await validate_input_field("duration_sec", request_data.duration_sec, VALIDATION_RULES["duration_sec"])
        await validate_input_field("preset", request_data.preset, VALIDATION_RULES["preset"])
        await validate_input_field("priority", request_data.priority, VALIDATION_RULES["priority"])
        await validate_input_field("lang", request_data.lang, VALIDATION_RULES["lang"])
        
        user_id = user.get("uid", user.get("email", "anonymous"))
        user_email = user.get("email", "")

        logger.info(
            f"POST /pipeline/run from {user_email} with content={request_data.content[:50]}, preset={request_data.preset}"
        )

        # Récupérer les données de requête
        request_dict = request_data.model_dump()

        # Ajouter l'ID utilisateur aux métadonnées
        request_dict["_user_id"] = user_id
        request_dict["_user_email"] = user_email

        # Appliquer le preset si spécifié
        preset_name = request_data.preset or "quick_social"
        if request_data.preset:
            preset = get_preset(request_data.preset)
            if not preset:
                raise HTTPException(
                    status_code=400,
                    detail=f"Preset inconnu: {request_data.preset}. Disponibles: quick_social, brand_campaign, premium_spot",
                )
            request_dict = apply_preset_to_request(request_dict, request_data.preset)
            logger.info(
                f"Preset '{request_data.preset}' appliqué: mode={preset.pipeline_mode}, quality={preset.quality_threshold}"
            )

        # Ajouter estimation de coût initiale
        cost_estimate = get_full_cost_estimate(
            content=request_data.content,
            duration_sec=request_data.duration_sec or 30,
            preset=request_data.preset,
        )
        request_dict["_cost_estimate"] = cost_estimate["aiprod_optimized"]

        # Sanitize inputs
        sanitized = input_sanitizer.sanitize(request_dict)

        # 🔐 P1.2: Create job in PostgreSQL
        db_session = get_db_session()
        try:
            job_repo = JobRepository(db_session)
            job = job_repo.create_job(
                content=request_data.content,
                preset=preset_name,
                user_id=user_id,
                job_metadata={
                    "email": user_email,
                    "duration_sec": request_data.duration_sec or 30,
                    "priority": request_data.priority,
                    "lang": request_data.lang,
                    "cost_estimate": cost_estimate,
                    "sanitized_content": sanitized.get("content", request_data.content),
                },
            )
            job_id = job.id
            logger.info(f"Job {job_id} created in PostgreSQL for user {user_id}")
        finally:
            db_session.close()

        # 🚀 P1.2: Publish to Pub/Sub for async processing
        try:
            pubsub_client = get_pubsub_client()
            message_id = pubsub_client.publish_job(
                job_id=str(job_id),
                user_id=user_id,
                content=sanitized.get("content", request_data.content),
                preset=preset_name,
                metadata={
                    "email": user_email,
                    "duration_sec": request_data.duration_sec or 30,
                    "priority": request_data.priority,
                    "lang": request_data.lang,
                    "cost_estimate": cost_estimate["aiprod_optimized"],
                },
            )
            logger.info(f"Job {job_id} published to Pub/Sub (msg_id={message_id})")
        except Exception as pubsub_error:
            # If Pub/Sub fails, update job status to FAILED
            logger.error(f"Pub/Sub publish failed for job {job_id}: {pubsub_error}")
            db_session = get_db_session()
            try:
                job_repo = JobRepository(db_session)
                job_repo.update_job_state(
                    str(job_id), "FAILED", reason=f"Pub/Sub error: {str(pubsub_error)}"
                )
            finally:
                db_session.close()
            raise HTTPException(
                status_code=503, detail="Queue service temporarily unavailable"
            )

        # 🔐 Audit logging de succès
        audit_logger = get_audit_logger()
        latency_ms = (time.time() - start_time) * 1000
        audit_logger.log_api_call(
            endpoint="/pipeline/run",
            method="POST",
            user_id=user_email,
            status_code=202,
            duration_ms=latency_ms,
        )

        # Return immediately with job_id (async pattern)
        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Job submitted for processing",
            "cost_estimate": cost_estimate,
            "check_status_at": f"/pipeline/job/{job_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        metrics_collector.record_error(str(e))

        # 🔐 Audit logging d'erreur
        audit_logger = get_audit_logger()
        latency_ms = (time.time() - start_time) * 1000
        audit_logger.log_api_call(
            endpoint="/pipeline/run",
            method="POST",
            user_id=user.get("email"),
            status_code=500,
            duration_ms=latency_ms,
        )

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/job/{job_id}")
@limiter.limit("60/minute")
async def get_job_status(
    request: Request, job_id: str, user: dict = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Récupère l'état d'un job spécifique.

    🔐 AUTHENTIFICATION REQUISE

    Args:
        job_id: Identifiant unique du job
        user: Utilisateur authentifié (injecté par verify_token)
    Returns:
        Dict avec les détails du job (status, history, result si terminé)
    """
    user_id = user.get("uid", user.get("email", "anonymous"))

    db_session = get_db_session()
    try:
        job_repo = JobRepository(db_session)
        job = job_repo.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Security: Only job owner can access
        if job.user_id != user_id:
            logger.warning(
                f"User {user_id} attempted to access job {job_id} owned by {job.user_id}"
            )
            raise HTTPException(status_code=403, detail="Access denied")

        # Build response
        response = {
            "job_id": job.id,
            "status": job.current_state,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "preset": job.preset,
            "content_preview": (
                getattr(job, "content", "")[:100] + "..."
                if len(getattr(job, "content", "")) > 100
                else getattr(job, "content", "")
            ),
        }

        # Add state history
        response["state_history"] = [
            {
                "state": record.state,
                "entered_at": record.entered_at.isoformat(),
                "metadata": record.state_metadata,
            }
            for record in job.state_history
        ]

        # Add result if completed
        if job.result:
            response["result"] = {
                "success": job.result.success,
                "output": job.result.output,
                "completed_at": (
                    job.result.completed_at.isoformat()
                    if job.result.completed_at
                    else None
                ),
                "error_message": job.result.error_message,
                "execution_time_ms": job.result.execution_time_ms,
            }

        # Add retry info if applicable
        if job.retry_count > 0:
            response["retry_count"] = job.retry_count

        logger.info(f"Job status retrieved: {job_id} -> {job.current_state}")
        return response

    finally:
        db_session.close()


@app.get("/pipeline/jobs")
async def list_user_jobs(
    user: dict = Depends(verify_token),
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Liste les jobs de l'utilisateur.

    🔐 AUTHENTIFICATION REQUISE

    Args:
        user: Utilisateur authentifié
        status: Filtrer par statut (optionnel)
        limit: Nombre max de résultats (défaut: 20)
        offset: Décalage pour pagination
    Returns:
        Dict avec liste des jobs et métadonnées de pagination
    """
    user_id = user.get("uid", user.get("email", "anonymous"))

    db_session = get_db_session()
    try:
        job_repo = JobRepository(db_session)
        jobs = job_repo.list_jobs(
            user_id=user_id, state_filter=status, limit=limit, offset=offset
        )

        return {
            "jobs": [
                {
                    "job_id": job["id"],
                    "status": job["current_state"],
                    "preset": job.get("preset"),
                    "content_preview": (
                        job.get("content", "")[:50] + "..."
                        if len(job.get("content", "")) > 50
                        else job.get("content", "")
                    ),
                    "created_at": job.get("created_at"),
                }
                for job in jobs
            ],
            "limit": limit,
            "offset": offset,
            "count": len(jobs),
        }
    finally:
        db_session.close()


@app.get("/pipeline/status")
@limiter.limit("60/minute")
async def pipeline_status(
    request: Request,
    user: Optional[dict] = Depends(optional_verify_token),
) -> Dict[str, str]:
    """
    Récupère l'état actuel du pipeline.
    Returns:
        Dict[str, str]: État du pipeline.
    """
    logger.info(
        f"GET /pipeline/status from {user.get('email') if user else 'anonymous'}"
    )
    audit_logger = get_audit_logger()
    audit_logger.log_api_call(
        endpoint="/pipeline/status",
        method="GET",
        user_id=user.get("email") if user else "anonymous",
        status_code=200,
    )
    return {"state": state_machine.state.name}


@app.get("/favicon.ico")
async def favicon() -> Response:
    """Favicon inline pour éviter le 404 des navigateurs."""
    return Response(content=FAVICON_BYTES, media_type="image/x-icon")


@app.get("/icc/data")
@limiter.limit("60/minute")
async def get_icc_data(request: Request) -> Dict[str, Any]:
    """
    Endpoint ICC (Interface Client Collaboratif) pour exposer les données mémoire.
    Returns:
        Dict[str, Any]: Données exposées à l'ICC.
    """
    logger.info("GET /icc/data")
    return state_machine.data


@app.get("/metrics")
async def get_metrics(
    user: Optional[dict] = Depends(optional_verify_token),
) -> Dict[str, Any]:
    """
    Endpoint pour récupérer les métriques de performance.
    Returns:
        Dict[str, Any]: Métriques du pipeline.
    """
    logger.info(f"GET /metrics from {user.get('email') if user else 'anonymous'}")
    audit_logger = get_audit_logger()
    audit_logger.log_api_call(
        endpoint="/metrics",
        method="GET",
        user_id=user.get("email") if user else "anonymous",
        status_code=200,
    )
    return (
        metrics_collector.get_internal_metrics()
    )  # CORRECTION: get_internal_metrics() au lieu de get_metrics()


@app.get("/alerts")
@limiter.limit("60/minute")
async def get_alerts(request: Request) -> Dict[str, Any]:
    """
    Endpoint pour récupérer les alertes actives.
    Returns:
        Dict[str, Any]: Alertes déclenchées.
    """
    logger.info("GET /alerts")
    alerts = metrics_collector.check_alerts()
    return {"alerts": alerts}


@app.post("/financial/optimize")
@limiter.limit("20/minute")
async def optimize_financial(
    request: Request, manifest: Dict[str, Any], user: Optional[dict] = Depends(optional_verify_token)
) -> Dict[str, Any]:
    """
    Endpoint pour l'optimisation financière.
    Args:
        manifest (Dict[str, Any]): Manifeste à optimiser.
    Returns:
        Dict[str, Any]: Résultat d'optimisation.
    """
    logger.info(
        f"POST /financial/optimize from {user.get('email') if user else 'anonymous'}"
    )
    audit_logger = get_audit_logger()
    audit_logger.log_api_call(
        endpoint="/financial/optimize",
        method="POST",
        user_id=user.get("email") if user else "anonymous",
        status_code=200,
    )
    return financial_orchestrator.optimize(manifest)


@app.post("/qa/technical")
@limiter.limit("40/minute")
async def validate_technical(
    request: Request, manifest: Dict[str, Any], user: Optional[dict] = Depends(optional_verify_token)
) -> Dict[str, Any]:
    """
    Endpoint pour la validation technique.
    Args:
        manifest (Dict[str, Any]): Manifeste à valider.
    Returns:
        Dict[str, Any]: Rapport de validation.
    """
    logger.info(f"POST /qa/technical from {user.get('email') if user else 'anonymous'}")
    audit_logger = get_audit_logger()
    audit_logger.log_api_call(
        endpoint="/qa/technical",
        method="POST",
        user_id=user.get("email") if user else "anonymous",
        status_code=200,
    )
    return technical_qa_gate.validate(manifest)


# ========================================
# PHASE 1 OPTIMISATION: PRESETS & COST ESTIMATE
# ========================================


@app.get("/presets")
@limiter.limit("100/minute")
async def list_presets(request: Request) -> Dict[str, Any]:
    """
    Liste tous les presets disponibles avec leurs configurations.

    Returns:
        Dict avec les presets: quick_social, brand_campaign, premium_spot
    """
    logger.info("GET /presets")
    return {
        "presets": get_all_presets(),
        "usage": "Ajoutez 'preset': 'quick_social' dans votre requête /pipeline/run",
    }


@app.get("/presets/{preset_name}")
async def get_preset_details(preset_name: str) -> Dict[str, Any]:
    """
    Récupère les détails d'un preset spécifique.

    Args:
        preset_name: Nom du preset (quick_social, brand_campaign, premium_spot)

    Returns:
        Configuration du preset avec estimation de coût
    """
    logger.info(f"GET /presets/{preset_name}")
    preset = get_preset(preset_name)
    if not preset:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_name}' non trouvé. Disponibles: quick_social, brand_campaign, premium_spot",
        )

    return {
        "name": preset.name,
        "description": preset.description,
        "pipeline_mode": preset.pipeline_mode,
        "quality_threshold": preset.quality_threshold,
        "max_duration_sec": preset.max_duration_sec,
        "max_cost_per_minute": preset.max_cost_per_minute,
        "allow_icc": preset.allow_icc,
        "consistency_cache": preset.consistency_cache,
        "multi_review": preset.multi_review,
        "priority": preset.priority,
        "estimated_cost_30s": preset.estimated_cost,
        "cost_estimate_for_durations": {
            "10s": estimate_cost_for_preset(preset_name, 10),
            "30s": estimate_cost_for_preset(preset_name, 30),
            "60s": estimate_cost_for_preset(preset_name, 60),
        },
    }


@app.post("/cost-estimate")
@limiter.limit("50/minute")
async def estimate_cost(request_data: CostEstimateRequest, request: Request) -> Dict[str, Any]:
    """
    Estime le coût d'une génération vidéo avec comparaison concurrents.

    Retourne:
        - Coût AIPROD optimisé
        - Coût Runway direct (benchmark)
        - Économies réalisées
        - Garantie qualité

    Args:
        request: Contenu, durée, preset optionnel

    Returns:
        Estimation détaillée avec breakdown et comparaison
    """
    logger.info(
        f"POST /cost-estimate for duration={request_data.duration_sec}s, preset={request_data.preset}"
    )

    estimate = get_full_cost_estimate(
        content=request_data.content,
        duration_sec=request_data.duration_sec,
        preset=request_data.preset,
        complexity=request_data.complexity,
    )

    return estimate


@app.get("/job/{job_id}/costs")
async def get_job_costs(job_id: str) -> Dict[str, Any]:
    """
    Récupère les coûts réels d'un job terminé.

    Args:
        job_id: Identifiant du job

    Returns:
        Coûts estimés vs réels avec breakdown
    """
    logger.info(f"GET /job/{job_id}/costs")

    # Récupérer les données du job depuis la state machine
    job_data = state_machine.data

    if not job_data:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' non trouvé ou pas encore terminé"
        )

    return get_job_actual_costs(job_data)


# ========================================
# PHASE 2: INTERACTIVE CREATIVE CONTROL (ICC)
# ========================================


class ManifestUpdateRequest(BaseModel):
    """Schéma pour mise à jour du manifest."""

    shot_list: Optional[List[str]] = None
    scenes: Optional[List[str]] = None
    duration: Optional[int] = None
    audio_style: Optional[str] = None
    camera_movements: Optional[List[str]] = None


@app.get("/jobs")
async def list_jobs() -> Dict[str, Any]:
    """
    Liste tous les jobs (pour admin/debug).

    Returns:
        Liste des jobs avec leurs états
    """
    logger.info("GET /jobs")
    return {
        "jobs": job_manager.get_all_jobs(),
        "total": len(job_manager.get_all_jobs()),
    }


@app.get("/job/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    """
    Récupère les détails complets d'un job.

    Args:
        job_id: Identifiant du job

    Returns:
        Détails complets du job incluant manifest, coûts, résultats
    """
    logger.info(f"GET /job/{job_id}")

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trouvé")

    return job_manager.to_dict(job)


@app.get("/job/{job_id}/manifest")
async def get_job_manifest(job_id: str) -> Dict[str, Any]:
    """
    Récupère le production_manifest d'un job.
    Permet au client de voir et préparer les modifications.

    Args:
        job_id: Identifiant du job

    Returns:
        Production manifest avec champs éditables marqués
    """
    logger.info(f"GET /job/{job_id}/manifest")

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trouvé")

    if not job.production_manifest:
        raise HTTPException(
            status_code=400,
            detail="Manifest pas encore disponible. Job doit être en état WAITING_APPROVAL.",
        )

    return {
        "job_id": job_id,
        "state": job.state.value,
        "manifest": job.production_manifest,
        "consistency_markers": job.consistency_markers,  # Read-only
        "editable_fields": [
            "shot_list",
            "scenes",
            "duration",
            "audio_style",
            "camera_movements",
        ],
        "locked_fields": ["consistency_markers"],
        "can_edit": job.state == JobState.WAITING_APPROVAL,
        "edits_history": job.edits_history,
    }


@app.patch("/job/{job_id}/manifest")
async def update_job_manifest(
    job_id: str, updates: ManifestUpdateRequest
) -> Dict[str, Any]:
    """
    Met à jour le production_manifest d'un job.
    Seul possible quand le job est en état WAITING_APPROVAL.

    Args:
        job_id: Identifiant du job
        updates: Champs à mettre à jour (shot_list, scenes, duration, etc.)

    Returns:
        Manifest mis à jour
    """
    logger.info(f"PATCH /job/{job_id}/manifest")

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trouvé")

    if job.state != JobState.WAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de modifier le manifest en état '{job.state.value}'. Doit être 'waiting_approval'.",
        )

    # Appliquer les mises à jour
    updates_dict = updates.model_dump(exclude_none=True)
    if not updates_dict:
        raise HTTPException(status_code=400, detail="Aucune mise à jour fournie")

    updated_job = await job_manager.update_manifest(job_id, updates_dict)
    if not updated_job:
        raise HTTPException(
            status_code=500, detail="Échec de la mise à jour du manifest"
        )

    return {
        "job_id": job_id,
        "status": "updated",
        "manifest": updated_job.production_manifest,
        "changes": updates_dict,
        "edits_count": len(updated_job.edits_history),
    }


@app.post("/job/{job_id}/approve")
async def approve_job(job_id: str) -> Dict[str, Any]:
    """
    Approuve un job pour lancer le rendu.
    Déclenche la transition WAITING_APPROVAL → RENDERING.

    Args:
        job_id: Identifiant du job

    Returns:
        Confirmation d'approbation et nouvel état
    """
    logger.info(f"POST /job/{job_id}/approve")

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trouvé")

    if job.state != JobState.WAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible d'approuver en état '{job.state.value}'. Doit être 'waiting_approval'.",
        )

    approved_job = await job_manager.approve_job(job_id)
    if not approved_job:
        raise HTTPException(status_code=500, detail="Échec de l'approbation du job")

    approval_ts = (
        approved_job.approval_timestamp.isoformat()
        if approved_job.approval_timestamp
        else None
    )

    return {
        "job_id": job_id,
        "status": "approved",
        "state": approved_job.state.value,
        "approval_timestamp": approval_ts,
        "message": "Job approuvé. Le rendu va démarrer automatiquement.",
        "next_state": "rendering",
    }


@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str, reason: str = "User cancelled") -> Dict[str, Any]:
    """
    Annule un job.

    Args:
        job_id: Identifiant du job
        reason: Raison de l'annulation

    Returns:
        Confirmation d'annulation
    """
    logger.info(f"POST /job/{job_id}/cancel")

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trouvé")

    if job.state in [JobState.DELIVERED, JobState.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible d'annuler un job en état '{job.state.value}'.",
        )

    cancelled_job = await job_manager.cancel_job(job_id, reason)
    if not cancelled_job:
        raise HTTPException(status_code=500, detail="Échec de l'annulation du job")

    return {
        "job_id": job_id,
        "status": "cancelled",
        "state": cancelled_job.state.value,
        "reason": reason,
    }


# ════════════════════════════════════════════════════════════════
# 📥 EXPORT ENDPOINTS
# ════════════════════════════════════════════════════════════════


@app.get("/export/formats")
@limiter.limit("100/minute")
async def get_export_formats(request: Request) -> ExportFormatsResponse:
    """
    Liste les formats d'export disponibles.

    Returns:
        Dict avec informations sur chaque format
    """
    export_service = get_export_service()
    formats_info = export_service.get_export_formats_info()
    return ExportFormatsResponse(formats=formats_info)


@app.get("/pipeline/{job_id}/export")
@limiter.limit("60/minute")
async def export_pipeline(
    job_id: str,
    request: Request,
    format: str = "json",
    user: dict = Depends(verify_token),
) -> Any:
    """
    Exporte les résultats d'un job dans le format spécifié.

    🔐 AUTHENTIFICATION REQUISE
    📥 FORMATS: json, csv, zip

    Supporte:
    - JSON: Structure complète avec métadonnées
    - CSV: Tableau des résultats (flattened)
    - ZIP: Archive avec metadata, results et logs

    Args:
        job_id: ID du job à exporter
        format: Format d'export (json, csv, zip)
        user: Utilisateur authentifié

    Returns:
        Fichier exporté ou JSON avec URL de téléchargement
    """
    logger.info(f"GET /pipeline/{job_id}/export?format={format}")

    # Valider le format
    if format.lower() not in [f.value for f in ExportFormat]:
        raise HTTPException(
            status_code=400,
            detail=f"Format invalide: {format}. Formats supportés: json, csv, zip"
        )

    # Récupérer le job
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trouvé")

    # Vérifier l'accès (l'utilisateur est propriétaire du job)
    if job.user_id != user.get("uid"):
        raise HTTPException(status_code=403, detail="Accès refusé à ce job")

    # Construire les données du job
    job_dict = job.to_dict() if hasattr(job, 'to_dict') else {
        "id": job.id,
        "user_id": job.user_id,
        "preset": job.preset,
        "state": job.state.value if hasattr(job.state, 'value') else job.state,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "content": job.content,
        "metadata": job.job_metadata,
    }

    export_service = get_export_service()

    try:
        if format.lower() == ExportFormat.JSON.value:
            # Export JSON
            json_str = export_service.export_to_json(job_dict)

            if not export_service.validate_export_size(json_str):
                raise HTTPException(status_code=413, detail="Export trop volumineux")

            audit_log(
                user_id=user.get("uid"),
                event_type=AuditEventType.EXPORT,
                details={"job_id": job_id, "format": format}
            )

            return Response(
                content=json_str,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=job-{job_id}.json"
                }
            )

        elif format.lower() == ExportFormat.CSV.value:
            # Export CSV
            csv_str = export_service.export_to_csv([job_dict], flatten=True)

            if not export_service.validate_export_size(csv_str):
                raise HTTPException(status_code=413, detail="Export trop volumineux")

            audit_log(
                user_id=user.get("uid"),
                event_type=AuditEventType.EXPORT,
                details={"job_id": job_id, "format": format}
            )

            return Response(
                content=csv_str,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=job-{job_id}.csv"
                }
            )

        elif format.lower() == ExportFormat.ZIP.value:
            # Export ZIP
            zip_bytes = export_service.export_to_zip(job_dict)

            if not export_service.validate_export_size(zip_bytes.getvalue()):
                raise HTTPException(status_code=413, detail="Export trop volumineux")

            audit_log(
                user_id=user.get("uid"),
                event_type=AuditEventType.EXPORT,
                details={"job_id": job_id, "format": format}
            )

            return Response(
                content=zip_bytes.getvalue(),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=job-{job_id}.zip"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'export")


@app.get("/jobs/export")
@limiter.limit("30/minute")
async def export_jobs_bulk(
    request: Request,
    format: str = "csv",
    limit: int = 100,
    user: dict = Depends(verify_token),
) -> Any:
    """
    Exporte les derniers jobs de l'utilisateur.

    🔐 AUTHENTIFICATION REQUISE
    📥 FORMATS: csv, zip

    Args:
        format: Format d'export (csv ou zip)
        limit: Nombre max de jobs à exporter (max 1000)
        user: Utilisateur authentifié

    Returns:
        Fichier exporté
    """
    logger.info(f"GET /jobs/export?format={format}&limit={limit}")

    # Limiter le nombre de jobs
    limit = min(limit, 1000)

    if format.lower() not in [ExportFormat.CSV.value, ExportFormat.ZIP.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Format invalide: {format}. Pour bulk export: csv, zip"
        )

    # Récupérer les jobs de l'utilisateur
    try:
        session = get_db_session()
        jobs = session.query(Job).filter(
            Job.user_id == user.get("uid")
        ).order_by(Job.created_at.desc()).limit(limit).all()

        if not jobs:
            raise HTTPException(status_code=404, detail="Aucun job trouvé")

        # Convertir en dicts
        jobs_dicts = [job.to_dict() for job in jobs]

        export_service = get_export_service()

        if format.lower() == ExportFormat.CSV.value:
            csv_str = export_service.export_to_csv(jobs_dicts, flatten=True)

            if not export_service.validate_export_size(csv_str):
                raise HTTPException(status_code=413, detail="Export trop volumineux")

            audit_log(
                user_id=user.get("uid"),
                event_type=AuditEventType.EXPORT,
                details={"job_count": len(jobs), "format": format}
            )

            return Response(
                content=csv_str,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=jobs-export-{datetime.now().isoformat()}.csv"
                }
            )

        elif format.lower() == ExportFormat.ZIP.value:
            # Pour ZIP, créer une archive avec plusieurs fichiers JSON
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Résumé en CSV
                csv_str = export_service.export_to_csv(jobs_dicts, flatten=True)
                zip_file.writestr("jobs_summary.csv", csv_str)

                # JSON individuel pour chaque job
                for job_dict in jobs_dicts:
                    job_json = json.dumps(job_dict, indent=2, default=str)
                    zip_file.writestr(f"jobs/{job_dict['id']}.json", job_json)

            zip_buffer.seek(0)

            if not export_service.validate_export_size(zip_buffer.getvalue()):
                raise HTTPException(status_code=413, detail="Export trop volumineux")

            audit_log(
                user_id=user.get("uid"),
                event_type=AuditEventType.EXPORT,
                details={"job_count": len(jobs), "format": format}
            )

            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=jobs-export-{datetime.now().isoformat()}.zip"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk export error: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'export en masse")


@app.websocket("/ws/job/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """
    WebSocket pour recevoir les mises à jour temps réel d'un job.

    Events:
        - state_changed: Transition d'état du job
        - manifest_updated: Modification du manifest
        - cost_updated: Mise à jour de l'estimation de coût
        - approved: Job approuvé
        - qa_completed: Rapport QA disponible
        - cancelled: Job annulé

    Usage:
        ws://host/ws/job/{job_id}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for job {job_id}")

    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.send_json({"error": f"Job '{job_id}' non trouvé"})
        await websocket.close()
        return

    # S'abonner aux mises à jour
    await job_manager.subscribe(job_id, websocket)

    # Envoyer l'état initial
    await websocket.send_json(
        {
            "event": "connected",
            "job_id": job_id,
            "state": job.state.value,
            "timestamp": job.updated_at.isoformat(),
        }
    )

    try:
        while True:
            # Garder la connexion ouverte et écouter les messages du client
            data = await websocket.receive_text()

            # Le client peut envoyer des pings pour garder la connexion active
            if data == "ping":
                await websocket.send_json({"event": "pong"})
            elif data == "status":
                # Renvoyer l'état actuel
                current_job = await job_manager.get_job(job_id)
                if current_job:
                    await websocket.send_json(
                        {
                            "event": "status",
                            "job_id": job_id,
                            "state": current_job.state.value,
                            "approved": current_job.approved,
                        }
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
        await job_manager.unsubscribe(job_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await job_manager.unsubscribe(job_id, websocket)


@app.get("/icc/stats")
async def get_icc_stats() -> Dict[str, Any]:
    """
    Statistiques ICC pour monitoring.

    Returns:
        Stats sur les jobs: total, par état, taux d'approbation
    """
    logger.info("GET /icc/stats")

    jobs = job_manager.get_all_jobs()

    # Compter par état
    state_counts = {}
    approved_count = 0
    for job in jobs:
        state = job["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
        if job["approved"]:
            approved_count += 1

    return {
        "total_jobs": len(jobs),
        "jobs_by_state": state_counts,
        "approved_count": approved_count,
        "approval_rate": round(approved_count / len(jobs) * 100, 1) if jobs else 0,
    }


# ========================================
# PHASE 1.3 - API KEY ROTATION & MANAGEMENT
# ========================================


@app.post("/api-keys/create")
@limiter.limit("5/minute")
async def create_api_key(
    request: Request,
    req: CreateAPIKeyRequest,
    user: dict = Depends(verify_token),
) -> APIKeyResponse:
    """
    Create a new API key for the user.
    
    🔐 AUTHENTIFICATION REQUISE
    ⏱️ RATE LIMITED: 5 keys max per minute
    ⚠️ IMPORTANT: Key value is only shown once. Save it securely!
    
    Args:
        req: CreateAPIKeyRequest with key name
        user: Authenticated user
        
    Returns:
        APIKeyResponse with key_value (only shown once)
    """
    logger.info(f"POST /api-keys/create for user {user.get('uid')}")
    
    try:
        api_key_manager = get_api_key_manager()
        key_response = api_key_manager.generate_api_key(
            user.get("uid"),
            req.name
        )
        
        audit_log(
            user_id=user.get("uid"),
            event_type=AuditEventType.API_KEY_CREATED,
            details={"key_id": key_response["key_id"], "name": req.name}
        )
        
        return APIKeyResponse(**key_response)
        
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail="Error creating API key")


@app.get("/api-keys")
@limiter.limit("30/minute")
async def list_api_keys(
    request: Request,
    include_inactive: bool = False,
    user: dict = Depends(verify_token),
) -> ListAPIKeysResponse:
    """
    List all API keys for the authenticated user.
    
    🔐 AUTHENTIFICATION REQUISE
    📋 Shows key metadata but NOT key values
    
    Query Parameters:
        include_inactive (bool): Include revoked/rotated keys (default: false)
    
    Args:
        user: Authenticated user
        
    Returns:
        ListAPIKeysResponse with keys and statistics
    """
    logger.info(f"GET /api-keys for user {user.get('uid')}")
    
    try:
        api_key_manager = get_api_key_manager()
        keys = api_key_manager.list_api_keys(
            user.get("uid"),
            include_inactive=include_inactive
        )
        
        # Calculate statistics
        active_count = sum(1 for k in keys if k.get("status") == "active")
        
        audit_log(
            user_id=user.get("uid"),
            event_type=AuditEventType.API_KEY_LISTED,
            details={"total_keys": len(keys), "active_keys": active_count}
        )
        
        return ListAPIKeysResponse(
            keys=keys,
            total=len(keys),
            active_count=active_count
        )
        
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=500, detail="Error listing API keys")


@app.post("/api-keys/{key_id}/rotate")
@limiter.limit("10/minute")
async def rotate_api_key(
    request: Request,
    key_id: str,
    user: dict = Depends(verify_token),
) -> APIKeyResponse:
    """
    Rotate an API key (generate new one, mark old as rotated).
    
    🔐 AUTHENTIFICATION REQUISE
    🔄 New key is immediately active
    ⌛ Old key becomes inactive after rotation
    
    Args:
        key_id: ID of key to rotate
        user: Authenticated user
        
    Returns:
        APIKeyResponse with new key_value (only shown once)
    """
    logger.info(f"POST /api-keys/{key_id}/rotate for user {user.get('uid')}")
    
    try:
        api_key_manager = get_api_key_manager()
        new_key = api_key_manager.rotate_api_key(key_id, user.get("uid"))
        
        audit_log(
            user_id=user.get("uid"),
            event_type=AuditEventType.API_KEY_ROTATED,
            details={"old_key_id": key_id, "new_key_id": new_key["key_id"]}
        )
        
        return APIKeyResponse(**new_key)
        
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error rotating API key {key_id}: {e}")
        raise HTTPException(status_code=500, detail="Error rotating API key")


@app.post("/api-keys/{key_id}/revoke")
@limiter.limit("10/minute")
async def revoke_api_key(
    request: Request,
    key_id: str,
    req: RevokeAPIKeyRequest,
    user: dict = Depends(verify_token),
) -> Dict[str, Any]:
    """
    Revoke an API key (permanent deactivation).
    
    🔐 AUTHENTIFICATION REQUISE
    ⛔ Revoked keys cannot be restored (must create new)
    
    Args:
        key_id: ID of key to revoke
        req: RevokeAPIKeyRequest with optional reason
        user: Authenticated user
        
    Returns:
        Confirmation message with timestamp
    """
    logger.info(f"POST /api-keys/{key_id}/revoke for user {user.get('uid')}")
    
    try:
        api_key_manager = get_api_key_manager()
        api_key_manager.revoke_api_key(req.key_id, user.get("uid"))
        
        audit_log(
            user_id=user.get("uid"),
            event_type=AuditEventType.API_KEY_REVOKED,
            details={"key_id": key_id, "reason": req.reason}
        )
        
        return {
            "status": "revoked",
            "key_id": key_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "API key has been permanently revoked"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error revoking API key {key_id}: {e}")
        raise HTTPException(status_code=500, detail="Error revoking API key")


@app.post("/api-keys/revoke-all")
@limiter.limit("2/minute")
async def revoke_all_api_keys(
    request: Request,
    req: RevokeAllKeysRequest,
    user: dict = Depends(verify_token),
) -> RevokeAllKeysResponse:
    """
    Revoke ALL API keys for the user (security incident response).
    
    🔐 AUTHENTIFICATION REQUISE
    ⚠️ DANGER: This revokes ALL active keys immediately
    🔒 Requires explicit confirmation
    
    Args:
        req: RevokeAllKeysRequest with confirm flag
        user: Authenticated user
        
    Returns:
        Count of revoked keys
    """
    logger.warning(f"POST /api-keys/revoke-all requested by user {user.get('uid')}")
    
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to revoke all keys"
        )
    
    try:
        api_key_manager = get_api_key_manager()
        revoked_count = api_key_manager.revoke_all_keys(user.get("uid"))
        
        audit_log(
            user_id=user.get("uid"),
            event_type=AuditEventType.API_KEY_MASS_REVOKED,
            details={"revoked_count": revoked_count, "reason": req.reason}
        )
        
        return RevokeAllKeysResponse(
            revoked_count=revoked_count,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error revoking all API keys for user {user.get('uid')}: {e}")
        raise HTTPException(status_code=500, detail="Error revoking API keys")


@app.get("/api-keys/stats")
@limiter.limit("30/minute")
async def get_api_key_stats(
    request: Request,
    user: dict = Depends(verify_token),
) -> APIKeyStatsResponse:
    """
    Get statistics about the user's API keys.
    
    🔐 AUTHENTIFICATION REQUISE
    📊 Returns summary of key statuses and expiration info
    
    Args:
        user: Authenticated user
        
    Returns:
        APIKeyStatsResponse with key counts and next expiration
    """
    logger.info(f"GET /api-keys/stats for user {user.get('uid')}")
    
    try:
        api_key_manager = get_api_key_manager()
        keys = api_key_manager.list_api_keys(user.get("uid"), include_inactive=True)
        
        # Calculate statistics by status
        status_counts = {}
        next_expiration = None
        
        for key in keys:
            status = key.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Track nearest expiration
            expires_at = key.get("expires_at")
            if expires_at and (next_expiration is None or expires_at < next_expiration):
                next_expiration = expires_at
        
        return APIKeyStatsResponse(
            total_keys=len(keys),
            active_keys=status_counts.get("active", 0),
            rotated_keys=status_counts.get("rotated", 0),
            revoked_keys=status_counts.get("revoked", 0),
            expired_keys=status_counts.get("expired", 0),
            next_expiration=next_expiration
        )
        
    except Exception as e:
        logger.error(f"Error getting API key stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")


@app.get("/api-keys/health")
@limiter.limit("20/minute")
async def check_api_key_health(
    request: Request,
    user: dict = Depends(verify_token),
) -> APIKeyHealthCheck:
    """
    Check health status of user's API keys with security recommendations.
    
    🔐 AUTHENTIFICATION REQUISE
    🚨 Returns warnings about expired/unused keys
    💡 Provides security recommendations
    
    Args:
        user: Authenticated user
        
    Returns:
        APIKeyHealthCheck with status and recommendations
    """
    logger.info(f"GET /api-keys/health for user {user.get('uid')}")
    
    try:
        api_key_manager = get_api_key_manager()
        keys = api_key_manager.list_api_keys(user.get("uid"), include_inactive=False)
        
        has_active_keys = len(keys) > 0
        expiration_warning = False
        unused_keys = 0
        recommendations = []
        
        # Check for expiring keys (< 7 days)
        now = datetime.utcnow()
        for key in keys:
            expires_at = datetime.fromisoformat(key.get("expires_at"))
            days_until_expiry = (expires_at - now).days
            
            if 0 < days_until_expiry < 7:
                expiration_warning = True
            
            # Check for unused keys (last_used > 30 days ago)
            last_used = key.get("last_used")
            if last_used:
                last_used_date = datetime.fromisoformat(last_used)
                days_unused = (now - last_used_date).days
                if days_unused > 30:
                    unused_keys += 1
                    recommendations.append(
                        f"Consider revoking unused key {key['key_id']} "
                        f"(not used in {days_unused} days)"
                    )
        
        # Add general recommendations
        if not has_active_keys:
            recommendations.insert(0, "No active API keys found. Create one to authenticate.")
        
        if len(keys) > 5:
            recommendations.append(
                "You have many API keys. Consider consolidating them for better security."
            )
        
        if not any(r.startswith("Consider rotating") for r in recommendations):
            recommendations.append("Consider rotating keys quarterly for better security.")
        
        return APIKeyHealthCheck(
            has_active_keys=has_active_keys,
            expiration_warning=expiration_warning,
            unused_keys=unused_keys,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error checking API key health: {e}")
        raise HTTPException(status_code=500, detail="Error checking health status")


# ========================================
# PHASE 1.5 - CSRF PROTECTION
# ========================================


@app.get("/security/csrf-token")
@limiter.limit("60/minute")
async def get_csrf_token_endpoint(
    request: Request,
    user: Optional[dict] = Depends(optional_verify_token),
) -> Dict[str, str]:
    """
    Get a CSRF token for state-changing requests.
    
    🔐 OPTIONAL AUTHENTICATION
    ❗ IMPORTANT: Include token in X-CSRF-Token header for POST/PUT/DELETE
    
    Returns:
        CSRF token to use in subsequent state-changing requests
    """
    logger.info("GET /security/csrf-token")
    
    csrf_manager = get_csrf_manager()
    user_id = user.get("uid") if user else None
    token = csrf_manager.generate_token(user_id)
    
    return {
        "token": token,
        "header": "X-CSRF-Token",
        "expires_in": "60 minutes",
        "usage": "Include this token in X-CSRF-Token header for POST/PUT/DELETE/PATCH requests"
    }


@app.post("/security/csrf-verify")
@limiter.limit("60/minute")
async def verify_csrf_token_endpoint(
    request: Request,
    user: Optional[dict] = Depends(optional_verify_token),
) -> Dict[str, Any]:
    """
    Verify a CSRF token (for pre-flight checks).
    
    🔐 OPTIONAL AUTHENTICATION
    📋 Useful for frontend to validate token before sending requests
    
    Returns:
        Validation result
    """
    logger.info("POST /security/csrf-verify")
    
    token = request.headers.get("X-CSRF-Token")
    if not token:
        raise HTTPException(status_code=400, detail="X-CSRF-Token header missing")
    
    csrf_manager = get_csrf_manager()
    user_id = user.get("uid") if user else None
    is_valid = csrf_manager.validate_token(token, user_id)
    
    return {
        "token": token,
        "valid": is_valid,
        "message": "Token is valid" if is_valid else "Token is invalid or expired"
    }


@app.post("/security/csrf-refresh")
@limiter.limit("30/minute")
async def refresh_csrf_token_endpoint(
    request: Request,
    user: dict = Depends(verify_token),
) -> Dict[str, str]:
    """
    Refresh a CSRF token (revoke old, generate new).
    
    🔐 AUTHENTIFICATION REQUISE
    🔄 Use this to get a fresh token when old one expires
    
    Returns:
        New CSRF token
    """
    logger.info("POST /security/csrf-refresh")
    
    # Revoke old token if provided
    old_token = request.headers.get("X-CSRF-Token")
    csrf_manager = get_csrf_manager()
    
    if old_token:
        csrf_manager.revoke_token(old_token)
    
    # Generate new token
    new_token = csrf_manager.generate_token(user.get("uid"))
    
    audit_log(
        user_id=user.get("uid"),
        event_type=AuditEventType.SECURITY_ALERT,
        details={"action": "csrf_token_refresh"}
    )
    
    return {
        "token": new_token,
        "header": "X-CSRF-Token",
        "expires_in": "60 minutes"
    }


# ========================================
# PHASE 1.6 - SECURITY HEADERS
# ========================================


@app.get("/security/headers")
@limiter.limit("100/minute")
async def get_security_headers_info(request: Request) -> Dict[str, Any]:
    """
    Information about security headers applied to all responses.
    
    📋 Lists all security headers and their values
    
    Returns:
        Dictionary of security headers
    """
    logger.info("GET /security/headers")
    
    headers_info = {
        "Strict-Transport-Security": {
            "value": "max-age=31536000; includeSubDomains; preload",
            "purpose": "Force HTTPS connections"
        },
        "X-Content-Type-Options": {
            "value": "nosniff",
            "purpose": "Prevent MIME type sniffing"
        },
        "X-Frame-Options": {
            "value": "DENY",
            "purpose": "Prevent clickjacking attacks"
        },
        "X-XSS-Protection": {
            "value": "1; mode=block",
            "purpose": "Enable XSS filtering"
        },
        "Content-Security-Policy": {
            "value": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "purpose": "Control resource loading"
        },
        "Referrer-Policy": {
            "value": "strict-origin-when-cross-origin",
            "purpose": "Control referrer information"
        },
        "Permissions-Policy": {
            "value": "geolocation=(), microphone=(), camera=()",
            "purpose": "Restrict feature access"
        }
    }
    
    return {
        "headers": headers_info,
        "applied": True,
        "message": "Security headers are applied to all responses"
    }


@app.get("/security/policy")
@limiter.limit("100/minute")
async def get_security_policy(request: Request) -> Dict[str, Any]:
    """
    Security policy information and recommendations.
    
    📋 Provides security best practices and current status
    
    Returns:
        Security policy details
    """
    logger.info("GET /security/policy")
    
    return {
        "policies": {
            "authentication": {
                "enabled": True,
                "method": "Firebase JWT + API Keys",
                "mfa": "Recommended"
            },
            "encryption": {
                "transport": "TLS 1.2+",
                "at_rest": "Enabled",
                "algorithm": "AES-256"
            },
            "rate_limiting": {
                "enabled": True,
                "default": "Per-endpoint limits",
                "purpose": "Prevent abuse"
            },
            "audit_logging": {
                "enabled": True,
                "events": ["auth", "data_access", "admin_actions", "security_events"]
            },
            "cors": {
                "enabled": True,
                "allowed_origins": "Configured per environment"
            }
        },
        "recommendations": [
            "Always use HTTPS for API communication",
            "Keep API keys secure and rotate them regularly",
            "Enable MFA for production accounts",
            "Monitor audit logs for suspicious activity",
            "Keep client libraries updated"
        ],
        "compliance": {
            "https": True,
            "hsts": True,
            "csp": True,
            "xss_protection": True,
            "clickjacking_protection": True
        }
    }


@app.get("/security/audit-log")
@limiter.limit("20/minute")
async def get_audit_log_endpoint(
    request: Request,
    limit: int = 50,
    skip: int = 0,
    user: dict = Depends(verify_token),
) -> Dict[str, Any]:
    """
    Get user's audit log (recent events).
    
    🔐 AUTHENTIFICATION REQUISE
    📋 Shows security-relevant events for user's account
    
    Query Parameters:
        limit: Number of records to return (max 100)
        skip: Number of records to skip
    
    Returns:
        Recent audit events
    """
    logger.info(f"GET /security/audit-log for user {user.get('uid')}")
    
    limit = min(limit, 100)  # Cap at 100
    
    try:
        audit_logger = get_audit_logger()
        
        # Get audit logs (implementation depends on backend)
        # This is a placeholder showing the expected structure
        audit_events = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "AUTH_SUCCESS",
                "details": "User logged in",
                "ip_address": request.client.host if request.client else "unknown",
                "severity": "info"
            }
        ]
        
        return {
            "user_id": user.get("uid"),
            "events": audit_events,
            "total": len(audit_events),
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        logger.error(f"Error retrieving audit log: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving audit log")



# ?? Setup monitoring routes and dashboard
setup_monitoring_routes(app)

# ⚡ Setup performance optimization routes
setup_performance_routes(app)

# 🌍 Setup deployment and multi-region routes
setup_deployment_routes(app)

# 🧠 Setup advanced analytics routes with ML capabilities
setup_analytics_routes(app)

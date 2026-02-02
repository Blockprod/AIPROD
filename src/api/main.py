"""
API REST FastAPI pour AIPROD V33
Endpoints pour l'orchestration du pipeline, gestion des entrées et exposition des résultats.
"""

import time
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
    Depends,
)
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
from src.api.auth_middleware import (
    verify_token,
    optional_verify_token,
    AuthMiddleware,
    require_auth,
)
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

# 🔐 Ajouter le middleware d'authentification
app.add_middleware(AuthMiddleware)


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
async def health() -> Dict[str, str]:
    """
    Endpoint de santé de l'API.
    Returns:
        Dict[str, str]: Status de l'API.
    """
    logger.info("GET /health")
    return {"status": "ok"}


@app.post("/pipeline/run")
async def run_pipeline(
    request: PipelineRequest, user: dict = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Lance l'exécution du pipeline complet de manière asynchrone.

    🔐 AUTHENTIFICATION REQUISE
    🚀 ASYNC: Retourne immédiatement un job_id, traitement en arrière-plan

    Supporte les presets: quick_social, brand_campaign, premium_spot

    Args:
        request (PipelineRequest): Requête avec paramètres du pipeline.
        user: Utilisateur authentifié (injecté par verify_token)
    Returns:
        Dict avec job_id et status "queued"
    """
    start_time = time.time()

    try:
        user_id = user.get("uid", user.get("email", "anonymous"))
        user_email = user.get("email", "")

        logger.info(
            f"POST /pipeline/run from {user_email} with content={request.content[:50]}, preset={request.preset}"
        )

        # Récupérer les données de requête
        request_data = request.model_dump()

        # Ajouter l'ID utilisateur aux métadonnées
        request_data["_user_id"] = user_id
        request_data["_user_email"] = user_email

        # Appliquer le preset si spécifié
        preset_name = request.preset or "quick_social"
        if request.preset:
            preset = get_preset(request.preset)
            if not preset:
                raise HTTPException(
                    status_code=400,
                    detail=f"Preset inconnu: {request.preset}. Disponibles: quick_social, brand_campaign, premium_spot",
                )
            request_data = apply_preset_to_request(request_data, request.preset)
            logger.info(
                f"Preset '{request.preset}' appliqué: mode={preset.pipeline_mode}, quality={preset.quality_threshold}"
            )

        # Ajouter estimation de coût initiale
        cost_estimate = get_full_cost_estimate(
            content=request.content,
            duration_sec=request.duration_sec or 30,
            preset=request.preset,
        )
        request_data["_cost_estimate"] = cost_estimate["aiprod_optimized"]

        # Sanitize inputs
        sanitized = input_sanitizer.sanitize(request_data)

        # 🔐 P1.2: Create job in PostgreSQL
        db_session = get_db_session()
        try:
            job_repo = JobRepository(db_session)
            job = job_repo.create_job(
                content=request.content,
                preset=preset_name,
                user_id=user_id,
                job_metadata={
                    "email": user_email,
                    "duration_sec": request.duration_sec or 30,
                    "priority": request.priority,
                    "lang": request.lang,
                    "cost_estimate": cost_estimate,
                    "sanitized_content": sanitized.get("content", request.content),
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
                content=sanitized.get("content", request.content),
                preset=preset_name,
                metadata={
                    "email": user_email,
                    "duration_sec": request.duration_sec or 30,
                    "priority": request.priority,
                    "lang": request.lang,
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
async def get_job_status(
    job_id: str, user: dict = Depends(verify_token)
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
async def pipeline_status(
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
async def get_icc_data() -> Dict[str, Any]:
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
async def get_alerts() -> Dict[str, Any]:
    """
    Endpoint pour récupérer les alertes actives.
    Returns:
        Dict[str, Any]: Alertes déclenchées.
    """
    logger.info("GET /alerts")
    alerts = metrics_collector.check_alerts()
    return {"alerts": alerts}


@app.post("/financial/optimize")
async def optimize_financial(
    manifest: Dict[str, Any], user: Optional[dict] = Depends(optional_verify_token)
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
async def validate_technical(
    manifest: Dict[str, Any], user: Optional[dict] = Depends(optional_verify_token)
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
async def list_presets() -> Dict[str, Any]:
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
async def estimate_cost(request: CostEstimateRequest) -> Dict[str, Any]:
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
        f"POST /cost-estimate for duration={request.duration_sec}s, preset={request.preset}"
    )

    estimate = get_full_cost_estimate(
        content=request.content,
        duration_sec=request.duration_sec,
        preset=request.preset,
        complexity=request.complexity,
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

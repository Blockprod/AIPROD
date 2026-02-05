"""
AIPROD - Interactive Creative Control (ICC)
Endpoints pour le contrôle créatif interactif et la gestion des jobs.
"""
import uuid
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from src.utils.monitoring import logger


class JobState(str, Enum):
    """États possibles d'un job."""
    CREATED = "created"
    ANALYZING = "analyzing"
    CREATIVE_DIRECTION = "creative_direction"
    WAITING_APPROVAL = "waiting_approval"
    RENDERING = "rendering"
    QA_TECHNICAL = "qa_technical"
    QA_SEMANTIC = "qa_semantic"
    FINAL_APPROVAL = "final_approval"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Job:
    """Représente un job de génération vidéo."""
    id: str
    content: str
    preset: Optional[str]
    state: JobState
    created_at: datetime
    updated_at: datetime
    
    # Données de pipeline
    production_manifest: Optional[Dict[str, Any]] = None
    consistency_markers: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    render_result: Optional[Dict[str, Any]] = None
    qa_report: Optional[Dict[str, Any]] = None
    
    # Métadonnées
    priority: str = "low"
    lang: str = "en"
    brand_id: Optional[str] = None
    
    # ICC
    approved: bool = False
    approval_timestamp: Optional[datetime] = None
    edits_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # WebSocket subscribers
    _subscribers: List[Any] = field(default_factory=list, repr=False)


class JobManager:
    """
    Gestionnaire centralisé des jobs pour ICC.
    Permet la création, mise à jour et récupération des jobs.
    """
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
    
    async def create_job(
        self,
        content: str,
        preset: Optional[str] = None,
        priority: str = "low",
        lang: str = "en",
        brand_id: Optional[str] = None
    ) -> Job:
        """Crée un nouveau job."""
        async with self._lock:
            job_id = str(uuid.uuid4())[:8]
            now = datetime.now()
            
            job = Job(
                id=job_id,
                content=content,
                preset=preset,
                state=JobState.CREATED,
                created_at=now,
                updated_at=now,
                priority=priority,
                lang=lang,
                brand_id=brand_id
            )
            
            self._jobs[job_id] = job
            logger.info(f"JobManager: Created job {job_id}")
            return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Récupère un job par son ID."""
        return self._jobs.get(job_id)
    
    async def update_job_state(self, job_id: str, new_state: JobState) -> Optional[Job]:
        """Met à jour l'état d'un job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                old_state = job.state
                job.state = new_state
                job.updated_at = datetime.now()
                logger.info(f"JobManager: Job {job_id} state: {old_state} -> {new_state}")
                
                # Notifier les subscribers WebSocket
                await self._notify_subscribers(job, "state_changed", {
                    "old_state": old_state.value,
                    "new_state": new_state.value
                })
                
            return job
    
    async def update_manifest(
        self, 
        job_id: str, 
        manifest_updates: Dict[str, Any]
    ) -> Optional[Job]:
        """
        Met à jour le production_manifest d'un job.
        Seuls certains champs sont éditables (shot_list, duration, etc.)
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            if job.state != JobState.WAITING_APPROVAL:
                logger.warning(f"JobManager: Cannot edit manifest in state {job.state}")
                return None
            
            # Champs éditables
            editable_fields = ["shot_list", "scenes", "duration", "audio_style", "camera_movements"]
            
            # Merger les updates
            if job.production_manifest is None:
                job.production_manifest = {}
            
            changes = {}
            for field in editable_fields:
                if field in manifest_updates:
                    old_value = job.production_manifest.get(field)
                    job.production_manifest[field] = manifest_updates[field]
                    changes[field] = {"old": old_value, "new": manifest_updates[field]}
            
            # Enregistrer l'historique des modifications
            if changes:
                job.edits_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "changes": changes
                })
                job.updated_at = datetime.now()
                
                # Notifier
                await self._notify_subscribers(job, "manifest_updated", changes)
            
            logger.info(f"JobManager: Job {job_id} manifest updated: {list(changes.keys())}")
            return job
    
    async def set_manifest(
        self, 
        job_id: str, 
        manifest: Dict[str, Any],
        consistency_markers: Optional[Dict[str, Any]] = None
    ) -> Optional[Job]:
        """Définit le manifest complet (appelé par CreativeDirector)."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.production_manifest = manifest
                job.consistency_markers = consistency_markers
                job.updated_at = datetime.now()
                logger.info(f"JobManager: Job {job_id} manifest set")
            return job
    
    async def set_cost_estimate(
        self, 
        job_id: str, 
        cost_estimate: Dict[str, Any]
    ) -> Optional[Job]:
        """Définit l'estimation de coût."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.cost_estimate = cost_estimate
                job.updated_at = datetime.now()
                
                # Notifier
                await self._notify_subscribers(job, "cost_updated", cost_estimate)
                
            return job
    
    async def approve_job(self, job_id: str) -> Optional[Job]:
        """Approuve un job pour lancer le rendu."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            if job.state != JobState.WAITING_APPROVAL:
                logger.warning(f"JobManager: Cannot approve job in state {job.state}")
                return None
            
            job.approved = True
            job.approval_timestamp = datetime.now()
            job.state = JobState.RENDERING
            job.updated_at = datetime.now()
            
            # Notifier
            await self._notify_subscribers(job, "approved", {
                "timestamp": job.approval_timestamp.isoformat()
            })
            
            logger.info(f"JobManager: Job {job_id} approved, transitioning to RENDERING")
            return job
    
    async def set_render_result(
        self, 
        job_id: str, 
        render_result: Dict[str, Any]
    ) -> Optional[Job]:
        """Définit le résultat du rendu."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.render_result = render_result
                job.updated_at = datetime.now()
            return job
    
    async def set_qa_report(
        self, 
        job_id: str, 
        qa_report: Dict[str, Any]
    ) -> Optional[Job]:
        """Définit le rapport QA."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.qa_report = qa_report
                job.updated_at = datetime.now()
                
                # Notifier
                await self._notify_subscribers(job, "qa_completed", qa_report)
                
            return job
    
    async def cancel_job(self, job_id: str, reason: str = "User cancelled") -> Optional[Job]:
        """Annule un job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.state = JobState.CANCELLED
                job.updated_at = datetime.now()
                
                # Notifier
                await self._notify_subscribers(job, "cancelled", {"reason": reason})
                
                logger.info(f"JobManager: Job {job_id} cancelled: {reason}")
            return job
    
    async def subscribe(self, job_id: str, websocket: Any) -> bool:
        """Ajoute un subscriber WebSocket à un job."""
        job = self._jobs.get(job_id)
        if job:
            job._subscribers.append(websocket)
            logger.info(f"JobManager: WebSocket subscribed to job {job_id}")
            return True
        return False
    
    async def unsubscribe(self, job_id: str, websocket: Any) -> bool:
        """Retire un subscriber WebSocket d'un job."""
        job = self._jobs.get(job_id)
        if job and websocket in job._subscribers:
            job._subscribers.remove(websocket)
            logger.info(f"JobManager: WebSocket unsubscribed from job {job_id}")
            return True
        return False
    
    async def _notify_subscribers(
        self, 
        job: Job, 
        event_type: str, 
        data: Dict[str, Any]
    ) -> None:
        """Notifie tous les subscribers d'un job."""
        message = {
            "event": event_type,
            "job_id": job.id,
            "state": job.state.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        for subscriber in job._subscribers:
            try:
                await subscriber.send_json(message)
            except Exception as e:
                logger.warning(f"JobManager: Failed to notify subscriber: {e}")
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Retourne tous les jobs (pour admin/debug)."""
        return [
            {
                "id": job.id,
                "content": job.content[:50] + "..." if len(job.content) > 50 else job.content,
                "preset": job.preset,
                "state": job.state.value,
                "created_at": job.created_at.isoformat(),
                "approved": job.approved
            }
            for job in self._jobs.values()
        ]
    
    def to_dict(self, job: Job) -> Dict[str, Any]:
        """Convertit un job en dictionnaire."""
        return {
            "id": job.id,
            "content": job.content,
            "preset": job.preset,
            "state": job.state.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "production_manifest": job.production_manifest,
            "consistency_markers": job.consistency_markers,
            "cost_estimate": job.cost_estimate,
            "render_result": job.render_result,
            "qa_report": job.qa_report,
            "priority": job.priority,
            "lang": job.lang,
            "brand_id": job.brand_id,
            "approved": job.approved,
            "approval_timestamp": job.approval_timestamp.isoformat() if job.approval_timestamp else None,
            "edits_history": job.edits_history
        }


# Instance globale
_job_manager: Optional[JobManager] = None

def get_job_manager() -> JobManager:
    """Retourne l'instance globale du JobManager."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager

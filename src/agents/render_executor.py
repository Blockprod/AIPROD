"""
RenderExecutor Agent pour AIPROD V33 - Phase 3 Multi-Backend
Exécute le rendu complet: génération d'image + vidéo via:
- Runway ML (primary)
- Veo-3 via Vertex AI (premium)
- Replicate (fallback)
"""
import asyncio
import os
import time
from enum import Enum
from typing import Any, Dict, Optional, List
import httpx

try:
    from runwayml import RunwayML, TaskFailedError
    HAS_RUNWAY = True
except ImportError:
    HAS_RUNWAY = False
    RunwayML = None
    TaskFailedError = Exception

from src.utils.monitoring import logger
from src.utils.gcp_client import GCPClient


class VideoBackend(str, Enum):
    """Backends disponibles pour la génération vidéo."""
    RUNWAY = "runway"
    VEO3 = "veo3"
    REPLICATE = "replicate"
    AUTO = "auto"


class BackendConfig:
    """Configuration des backends avec priorité et fallback."""
    BACKEND_COSTS = {
        VideoBackend.RUNWAY: {"per_second": 5.0, "base": 5.0},  # ~30 credits pour 5s
        VideoBackend.VEO3: {"per_second": 0.50, "base": 0.10},  # ~$2.60 pour 5s
        VideoBackend.REPLICATE: {"per_second": 0.05, "base": 0.01},  # ~$0.26 pour 5s
    }
    
    BACKEND_QUALITY = {
        VideoBackend.RUNWAY: 0.95,  # Meilleure qualité
        VideoBackend.VEO3: 0.92,    # Très haute qualité
        VideoBackend.REPLICATE: 0.75,  # Qualité acceptable
    }
    
    FALLBACK_ORDER = [VideoBackend.RUNWAY, VideoBackend.VEO3, VideoBackend.REPLICATE]


class RenderExecutor:
    """
    Agent responsable de l'exécution du rendu multi-backend.
    Workflow: Image (gen4_image) → Vidéo (gen4_turbo/veo3/replicate) → GCS Upload
    """
    def __init__(self, preferred_backend: VideoBackend = VideoBackend.AUTO):
        # API Keys
        self.runway_api_key = os.getenv("RUNWAYML_API_SECRET") or os.getenv("RUNWAY_API_KEY", "")
        if self.runway_api_key:
            self.runway_api_key = self.runway_api_key.strip()
        
        # Check if runway is available
        if not HAS_RUNWAY and preferred_backend == VideoBackend.RUNWAY:
            logger.warning("RunwayML not installed. Falling back to alternative backends.")
            preferred_backend = VideoBackend.VEO3
        
        self.replicate_api_key = os.getenv("REPLICATE_API_TOKEN", "")
        self.gcp_project = os.getenv("GCP_PROJECT_ID", "aiprod-484120")
        
        # GCP Client
        self.gcp_client = GCPClient()
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "aiprod-484120-assets")
        
        # Backend selection
        self.preferred_backend = preferred_backend
        self._backend_health: Dict[VideoBackend, bool] = {
            VideoBackend.RUNWAY: True,
            VideoBackend.VEO3: True,
            VideoBackend.REPLICATE: True,
        }
        self._error_counts: Dict[VideoBackend, int] = {
            VideoBackend.RUNWAY: 0,
            VideoBackend.VEO3: 0,
            VideoBackend.REPLICATE: 0,
        }

    def _select_backend(
        self, 
        budget_remaining: Optional[float] = None,
        quality_required: float = 0.8,
        speed_priority: bool = False
    ) -> VideoBackend:
        """
        Sélectionne le meilleur backend basé sur:
        - Budget restant
        - Qualité requise
        - Priorité vitesse
        - Santé des backends
        """
        available_backends = [
            b for b in BackendConfig.FALLBACK_ORDER 
            if self._backend_health.get(b, False) and self._error_counts.get(b, 0) < 3
        ]
        
        if not available_backends:
            # Reset errors if all backends failed
            self._error_counts = {b: 0 for b in VideoBackend}
            available_backends = list(BackendConfig.FALLBACK_ORDER)
        
        # Si un backend spécifique est demandé
        if self.preferred_backend != VideoBackend.AUTO:
            if self.preferred_backend in available_backends:
                return self.preferred_backend
        
        # Stratégie de sélection basée sur le budget
        if budget_remaining is not None:
            # Budget TRÈS limité: choisir le moins cher, assouplir qualité
            if budget_remaining <= 1.0:
                quality_required = 0.7  # Tolérer Replicate (0.75)
                # Trier les backends par coût
                backend_costs = {}
                for b in available_backends:
                    cost = (BackendConfig.BACKEND_COSTS.get(b, {}).get("base", 100) + 
                           BackendConfig.BACKEND_COSTS.get(b, {}).get("per_second", 10) * 5)
                    backend_costs[b] = cost
                
                # Filtrer par qualité minimale
                quality_backends = [
                    b for b in available_backends
                    if BackendConfig.BACKEND_QUALITY.get(b, 0) >= quality_required
                ]
                
                if not quality_backends:
                    quality_backends = available_backends
                
                # Choisir le moins cher
                return min(quality_backends, key=lambda b: backend_costs[b])
            
            # Budget ÉLEVÉ: choisir la meilleure qualité affodable
            elif budget_remaining >= 50.0:
                # Choisir le backend avec meilleure qualité parmi les affodables
                backend_costs = {}
                for b in available_backends:
                    cost = (BackendConfig.BACKEND_COSTS.get(b, {}).get("base", 100) + 
                           BackendConfig.BACKEND_COSTS.get(b, {}).get("per_second", 10) * 5)
                    backend_costs[b] = cost
                
                # Filtrer par coût d'abord
                affordable_backends = [b for b in available_backends if backend_costs[b] <= budget_remaining]
                
                if not affordable_backends:
                    affordable_backends = available_backends
                
                # Parmi les affodables, choisir la meilleure qualité
                return max(affordable_backends, key=lambda b: BackendConfig.BACKEND_QUALITY.get(b, 0))
        
        # Mode par défaut: filtrer par qualité requise
        quality_backends = [
            b for b in available_backends
            if BackendConfig.BACKEND_QUALITY.get(b, 0) >= quality_required
        ]
        
        if not quality_backends:
            quality_backends = available_backends
        
        # Speed priority = Replicate (plus rapide mais qualité moindre)
        if speed_priority and VideoBackend.REPLICATE in quality_backends:
            return VideoBackend.REPLICATE
        
        # Par défaut, retourner le premier backend qualifié
        return quality_backends[0] if quality_backends else VideoBackend.RUNWAY

    async def run(
        self, 
        prompt_bundle: Dict[str, Any],
        backend: Optional[VideoBackend] = None,
        budget_remaining: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Workflow complet de rendu: Image → Vidéo → Upload GCS.
        Avec fallback automatique entre backends.
        """
        logger.info(f"RenderExecutor: Starting multi-backend rendering workflow")
        
        # Fallback mock si pas de clé API
        if not self.runway_api_key or self.runway_api_key == "your-runway-api-key":
            logger.warning("RenderExecutor: No API key, using mock")
            await asyncio.sleep(0.2)
            return {
                "status": "rendered_mock",
                "video_url": "gs://mock/video.mp4",
                "image_url": "gs://mock/image.jpg",
                "assets": {
                    "video": "gs://mock/video.mp4",
                    "image": "gs://mock/image.jpg"
                },
                "backend": "mock"
            }
        
        start_time = time.time()
        selected_backend = backend or self._select_backend(
            budget_remaining=budget_remaining,
            quality_required=prompt_bundle.get("quality_required", 0.8)
        )
        
        try:
            prompt = prompt_bundle.get("text_prompt", "A joyful scene")
            
            # 1️⃣ Générer image de concept (toujours Runway pour images)
            logger.info(f"RenderExecutor: Step 1/3 - Generating concept image...")
            concept_image_url = await self._generate_concept_image(prompt)
            
            # 2️⃣ Générer vidéo depuis l'image avec le backend sélectionné
            logger.info(f"RenderExecutor: Step 2/3 - Generating video with backend: {selected_backend.value}")
            video_url = await self._generate_video_with_fallback(
                concept_image_url, 
                prompt,
                selected_backend
            )
            
            # 3️⃣ Upload vers GCS
            logger.info(f"RenderExecutor: Step 3/3 - Uploading to GCS...")
            image_gcs_path = await self._upload_to_gcs(concept_image_url, asset_type="images")
            video_gcs_path = await self._upload_to_gcs(video_url, asset_type="videos")
            
            duration = time.time() - start_time
            
            # Report metrics
            await self._report_success_metrics(selected_backend, duration, prompt_bundle)
            
            result = {
                "status": "rendered",
                "video_url": video_gcs_path,
                "image_url": image_gcs_path,
                "assets": {
                    "video": video_gcs_path,
                    "image": image_gcs_path
                },
                "backend": selected_backend.value,
                "models": self._get_models_for_backend(selected_backend),
                "prompt": prompt,
                "duration_seconds": duration,
                "cost_estimate": self._estimate_cost(selected_backend, 5)
            }
            logger.info(f"RenderExecutor: Complete! Backend: {selected_backend.value}, Duration: {duration:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"RenderExecutor error: {e}")
            await self._report_error_metrics(selected_backend, str(e))
            return {"status": "error", "error": str(e), "backend": selected_backend.value}

    async def _generate_video_with_fallback(
        self, 
        image_url: str, 
        prompt: str,
        primary_backend: VideoBackend
    ) -> str:
        """Génère une vidéo avec fallback automatique entre backends."""
        backends_to_try = [primary_backend] + [
            b for b in BackendConfig.FALLBACK_ORDER 
            if b != primary_backend and self._backend_health.get(b, False)
        ]
        
        last_error = None
        for backend in backends_to_try:
            try:
                logger.info(f"RenderExecutor: Trying backend {backend.value}...")
                
                if backend == VideoBackend.RUNWAY:
                    return await self._generate_video_runway(image_url, prompt)
                elif backend == VideoBackend.VEO3:
                    return await self._generate_video_veo3(image_url, prompt)
                elif backend == VideoBackend.REPLICATE:
                    return await self._generate_video_replicate(image_url, prompt)
                    
            except Exception as e:
                logger.warning(f"RenderExecutor: Backend {backend.value} failed: {e}")
                self._error_counts[backend] = self._error_counts.get(backend, 0) + 1
                
                # Mark as unhealthy if too many errors
                if self._error_counts[backend] >= 3:
                    self._backend_health[backend] = False
                    logger.warning(f"RenderExecutor: Backend {backend.value} marked unhealthy")
                
                last_error = e
                continue
        
        raise Exception(f"All backends failed. Last error: {last_error}")

    async def _generate_video_runway(self, image_url: str, prompt: str) -> str:
        """Génère une vidéo avec Runway ML gen4_turbo."""
        return await self._generate_video_from_image(image_url, prompt)

    async def _generate_video_veo3(self, image_url: str, prompt: str) -> str:
        """
        Génère une vidéo avec Google Veo-3 via Vertex AI.
        Requires: google-cloud-aiplatform
        """
        try:
            import google.cloud.aiplatform as aiplatform  # type: ignore
            
            aiplatform.init(project=self.gcp_project, location="us-central1")
            
            # Note: Veo-3 API structure - adapté selon la documentation officielle
            video_prompt = f"Smooth cinematic camera movement: {prompt}"
            
            logger.info(f"RenderExecutor: Veo-3 prompt: {video_prompt[:80]}...")
            
            # Veo-3 endpoint call (structure peut varier selon version API)
            endpoint = aiplatform.Endpoint(
                endpoint_name=f"projects/{self.gcp_project}/locations/us-central1/publishers/google/models/veo-3"
            )
            
            response = endpoint.predict(
                instances=[{
                    "prompt": video_prompt,
                    "image": image_url,
                    "duration_seconds": 5,
                    "aspect_ratio": "16:9"
                }]
            )
            
            if response.predictions:
                video_url = response.predictions[0].get("video_uri")
                if video_url:
                    logger.info(f"RenderExecutor: Veo-3 video generated: {video_url[:50]}...")
                    return video_url
            
            raise Exception("No video URL in Veo-3 response")
            
        except ImportError:
            logger.warning("RenderExecutor: google-cloud-aiplatform not installed")
            raise Exception("Veo-3 requires google-cloud-aiplatform package")
        except Exception as e:
            logger.error(f"RenderExecutor: Veo-3 failed: {e}")
            raise

    async def _generate_video_replicate(self, image_url: str, prompt: str) -> str:
        """
        Génère une vidéo avec Replicate (Stable Video Diffusion ou autre).
        Fallback économique.
        """
        if not self.replicate_api_key:
            raise Exception("REPLICATE_API_TOKEN not configured")
        
        try:
            import replicate  # type: ignore
            
            video_prompt = f"Smooth cinematic camera movement: {prompt}"
            logger.info(f"RenderExecutor: Replicate prompt: {video_prompt[:80]}...")
            
            # Utiliser Stable Video Diffusion ou modèle similaire
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: replicate.run(
                    "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                    input={
                        "input_image": image_url,
                        "motion_bucket_id": 127,
                        "cond_aug": 0.02,
                        "decoding_t": 7,
                        "video_length": "25_frames_with_svd_xt"
                    }
                )
            )
            
            if output:
                video_url = str(output)
                logger.info(f"RenderExecutor: Replicate video generated: {video_url[:50]}...")
                return video_url
            
            raise Exception("No video URL in Replicate response")
            
        except ImportError:
            logger.warning("RenderExecutor: replicate package not installed")
            raise Exception("Replicate requires replicate package")
        except Exception as e:
            logger.error(f"RenderExecutor: Replicate failed: {e}")
            raise

    def _get_models_for_backend(self, backend: VideoBackend) -> Dict[str, str]:
        """Retourne les modèles utilisés pour un backend."""
        models = {
            VideoBackend.RUNWAY: {"image": "gen4_image", "video": "gen4_turbo"},
            VideoBackend.VEO3: {"image": "gen4_image", "video": "veo-3"},
            VideoBackend.REPLICATE: {"image": "gen4_image", "video": "stable-video-diffusion"},
        }
        return models.get(backend, {"image": "unknown", "video": "unknown"})

    def _estimate_cost(self, backend: VideoBackend, duration: int) -> float:
        """Estime le coût pour un backend donné."""
        costs = BackendConfig.BACKEND_COSTS.get(backend, {"base": 0, "per_second": 0})
        return costs["base"] + costs["per_second"] * duration

    async def _report_success_metrics(
        self, 
        backend: VideoBackend, 
        duration: float,
        prompt_bundle: Dict[str, Any]
    ) -> None:
        """Envoie les métriques de succès à Cloud Monitoring."""
        try:
            from src.utils.custom_metrics import get_metrics_collector, report_metric
            collector = get_metrics_collector()
            
            # Pipeline duration
            collector.report_metric("pipeline_duration", duration, {"backend": backend.value})
            
            # Cost
            cost = self._estimate_cost(backend, 5)
            collector.report_metric("cost_per_job", cost, {"backend": backend.value})
            
        except Exception as e:
            logger.warning(f"RenderExecutor: Failed to report metrics: {e}")

    async def _report_error_metrics(self, backend: VideoBackend, error: str) -> None:
        """Envoie les métriques d'erreur à Cloud Monitoring."""
        try:
            from src.utils.custom_metrics import report_error as metrics_report_error
            
            error_type = "runway_error" if backend == VideoBackend.RUNWAY else f"{backend.value}_error"
            metrics_report_error(error_type, backend=backend.value, details=error)
            
        except Exception as e:
            logger.warning(f"RenderExecutor: Failed to report error metrics: {e}")
    
    async def _generate_concept_image(self, prompt: str) -> str:
        """
        Génère une image de concept avec gen4_image.
        Cost: 5-8 credits par image
        """
        client = RunwayML(api_key=self.runway_api_key)
        
        # Améliorer le prompt pour l'image
        image_prompt = f"Professional cinematic high-quality image: {prompt}. 4K, detailed, vibrant colors, cinematic lighting."
        
        logger.info(f"RenderExecutor: Image prompt: {image_prompt[:80]}...")
        
        loop = asyncio.get_event_loop()
        
        try:
            # Create image task
            task = await loop.run_in_executor(
                None,
                lambda: client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=image_prompt,
                    ratio="1920:1080"
                )
            )
            
            logger.info(f"RenderExecutor: Image task created: {task.id}")
            
            # Wait for task completion
            completed_task = await loop.run_in_executor(
                None,
                lambda: task.wait_for_task_output()
            )
            
            logger.info(f"RenderExecutor: Concept image generated")
            
            # Extract URL from completed task
            task_output = getattr(completed_task, 'output', None)
            if task_output:
                image_url = self._extract_url(task_output)
                if image_url:
                    logger.info(f"RenderExecutor: Image URL extracted: {image_url[:80]}...")
                    return image_url
            
            raise Exception(f"No image URL in task output")
            
        except Exception as e:
            logger.error(f"RenderExecutor: Image generation failed: {e}")
            raise Exception(f"Image generation failed: {str(e)}")
    
    async def _generate_video_from_image(self, image_url: str, prompt: str) -> str:
        """
        Génère une vidéo basée sur l'image de concept avec gen4_turbo.
        Cost: 5 credits/second → 25 credits pour 5 secondes
        """
        client = RunwayML(api_key=self.runway_api_key)
        
        # Améliorer le prompt pour la vidéo
        video_prompt = f"Smooth cinematic camera movement, dynamic storytelling: {prompt}. Professional 4K video, smooth transitions."
        
        logger.info(f"RenderExecutor: Video prompt: {video_prompt[:80]}...")
        
        loop = asyncio.get_event_loop()
        
        try:
            # Create video task
            task = await loop.run_in_executor(
                None,
                lambda: client.image_to_video.create(
                    model="gen4_turbo",
                    prompt_image=image_url,
                    prompt_text=video_prompt,
                    ratio="1280:720",
                    duration=5
                )
            )
            
            logger.info(f"RenderExecutor: Video task created: {task.id}")
            
            # Wait for task completion
            completed_task = await loop.run_in_executor(
                None,
                lambda: task.wait_for_task_output()
            )
            
            logger.info(f"RenderExecutor: Video generated")
            
            # Extract URL from completed task
            task_output = getattr(completed_task, 'output', None)
            if task_output:
                video_url = self._extract_url(task_output)
                if video_url:
                    logger.info(f"RenderExecutor: Video URL extracted: {video_url[:80]}...")
                    return video_url
            
            raise Exception(f"No video URL in task output")
            
        except Exception as e:
            logger.error(f"RenderExecutor: Video generation failed: {e}")
            raise Exception(f"Video generation failed: {str(e)}")
    
    def _extract_url(self, output: Any) -> Optional[str]:
        """
        Extrait l'URL depuis différents formats de sortie.
        """
        if isinstance(output, dict):
            return output.get('url') or output.get('video_url') or output.get('image_url') or output.get('imageUrl') or output.get('videoUrl')
        elif isinstance(output, str):
            return output
        elif isinstance(output, list) and len(output) > 0:
            return str(output[0])
        return None
    
    async def _upload_to_gcs(self, media_url: str, asset_type: str = "videos") -> str:
        """
        Télécharge le média depuis Runway et l'upload vers GCS.
        asset_type: 'images' ou 'videos'
        """
        logger.info(f"RenderExecutor: Downloading {asset_type} from {media_url[:50]}...")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(media_url)
            response.raise_for_status()
            media_data = response.content
        
        timestamp = int(time.time())
        if asset_type == "images":
            ext = "jpg"
            local_filename = f"concept_{timestamp}.jpg"
            destination = f"{asset_type}/concept_{timestamp}.jpg"
        else:
            ext = "mp4"
            local_filename = f"render_{timestamp}.mp4"
            destination = f"{asset_type}/render_{timestamp}.mp4"
        
        # Save to local temp file
        local_path = f"/tmp/{local_filename}"
        logger.info(f"RenderExecutor: Saving to local: {local_path}")
        with open(local_path, "wb") as f:
            f.write(media_data)
        
        logger.info(f"RenderExecutor: Uploading to GCS: gs://{self.bucket_name}/{destination}")
        
        # Upload to GCS
        https_url = self.gcp_client.upload_to_storage(local_path, destination)
        
        logger.info(f"RenderExecutor: Upload complete: {https_url}")
        return https_url
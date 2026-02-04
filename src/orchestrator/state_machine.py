"""
StateMachine pour orchestrer le pipeline AIPROD V33 avec intégration des agents
"""
from enum import Enum, auto
from typing import Any, Dict
import asyncio
from src.utils.monitoring import logger
from src.agents.creative_director import CreativeDirector
from src.agents.fast_track_agent import FastTrackAgent
from src.agents.render_executor import RenderExecutor
from src.agents.semantic_qa import SemanticQA
from src.agents.visual_translator import VisualTranslator
from src.agents.supervisor import Supervisor
from src.agents.gcp_services_integrator import GoogleCloudServicesIntegrator
from src.agents.audio_generator import AudioGenerator
from src.agents.music_composer import MusicComposer

class PipelineState(Enum):
    INIT = auto()
    INPUT_SANITIZED = auto()
    AGENTS_EXECUTED = auto()
    QA_TECH = auto()
    QA_SEMANTIC = auto()
    FINAL_APPROVAL = auto()
    DELIVERED = auto()
    ERROR = auto()

class StateMachine:
    """
    Orchestrateur d'états pour le pipeline AIPROD V33.
    Gère les transitions, le retry et le logging.
    Intègre les agents principaux du pipeline.
    """
    def __init__(self):
        self.state = PipelineState.INIT
        self.data: Dict[str, Any] = {}
        self.retry_count = 0
        self.max_retries = 3
        # Instancie les agents
        self.creative_director = CreativeDirector()
        self.fast_track_agent = FastTrackAgent()
        self.render_executor = RenderExecutor()
        self.semantic_qa = SemanticQA()
        self.visual_translator = VisualTranslator()
        self.supervisor = Supervisor()
        self.gcp_services = GoogleCloudServicesIntegrator()
        # 🎤 Audio & Music Agents (Phase 1 Integration)
        self.audio_generator = AudioGenerator(tts_provider="auto")
        self.music_composer = MusicComposer()
        logger.info(f"StateMachine initialized in state {self.state}")
        logger.info("Audio & Music agents loaded successfully")

    def transition(self, new_state: PipelineState) -> None:
        logger.info(f"Transition: {self.state} -> {new_state}")
        self.state = new_state

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute le pipeline d'orchestration avec gestion des transitions, retry et agents.
        """
        try:
            self.transition(PipelineState.INPUT_SANITIZED)
            # Fast Track ou Creative Director selon priorité
            if inputs.get("priority") == "high":
                fast_output = await self.fast_track_agent.run(inputs)
                self.data["fast_track"] = fast_output
            else:
                fusion_output = await self.creative_director.run(inputs)
                self.data["fusion"] = fusion_output
            self.transition(PipelineState.AGENTS_EXECUTED)
            # Rendu des assets
            assets = self.data.get("fusion", {}).get("inputs") or self.data.get("fast_track", {}).get("inputs")
            render_output = await self.render_executor.run(assets)
            self.data["render"] = render_output
            
            # 🎤 Génération audio (voix humaine) — Phase 1.1
            logger.info("AudioGenerator: Starting TTS generation...")
            audio_manifest = self.data.get("fusion", {}) or self.data.get("fast_track", {})
            audio_manifest["lang"] = inputs.get("lang", "fr")
            audio_manifest["voice"] = inputs.get("voice", None)
            audio_manifest["emotion"] = inputs.get("emotion", None)
            audio_output = self.audio_generator.run(audio_manifest)
            self.data["audio"] = audio_output
            logger.info(f"AudioGenerator: TTS completed. Provider: {audio_output.get('provider')}")
            
            # 🎵 Génération musique — Phase 1.2
            logger.info("MusicComposer: Starting music generation...")
            music_manifest = audio_manifest.copy()
            music_manifest["music_style"] = inputs.get("music_style", "cinematic")
            music_manifest["mood"] = inputs.get("mood", None)
            music_manifest["duration"] = inputs.get("duration", 5)
            music_output = self.music_composer.run(music_manifest)
            self.data["music"] = music_output
            logger.info(f"MusicComposer: Music generated. Provider: {music_output.get('provider')}")
            
            # Validation sémantique
            self.transition(PipelineState.QA_SEMANTIC)
            semantic_report = await self.semantic_qa.run(render_output)
            self.data["semantic_qa"] = semantic_report
            # Traduction visuelle
            render_assets = render_output.get("assets", {})
            if render_assets:
                translated = await self.visual_translator.run(render_assets, target_lang=inputs.get("lang", "en"))
                self.data["visual_translation"] = translated
            else:
                logger.warning("No assets in render_output, skipping visual translation")
                self.data["visual_translation"] = {"status": "skipped", "reason": "no_assets"}
            # Approbation finale par Supervisor
            self.transition(PipelineState.FINAL_APPROVAL)
            supervisor_input = {
                "consistency_report": semantic_report,
                "cost_certification": self.data.get("cost_cert", {}),
                "technical_validation_report": self.data.get("tech_qa", {}),
                "quality_score": semantic_report.get("quality_score", 0.7),
                "client_budget": inputs.get("budget", 1000.0)
            }
            supervisor_result = await self.supervisor.run(supervisor_input)
            self.data["supervisor"] = supervisor_result
            # Intégration GCP Services pour livraison
            if supervisor_result.get("final_approval"):
                gcp_result = await self.gcp_services.run(supervisor_result.get("delivery_manifest", {}))
                self.data["gcp_delivery"] = gcp_result
            self.transition(PipelineState.DELIVERED)
            logger.info("Pipeline delivered successfully")
            return self.data
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.state = PipelineState.ERROR
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                logger.info(f"Retrying pipeline, attempt {self.retry_count}")
                return await self.run(inputs)
            else:
                logger.error("Max retries reached, aborting pipeline.")
                return {"error": str(e)}

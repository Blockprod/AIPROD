"""
Production Pipeline Orchestrator - State Machine with Checkpoint Integration
=============================================================================

Orchestrates production pipeline execution through a sovereign state machine.
Integrates checkpoint/resume capabilities for resilient execution.

Config: Charge AIPROD_V34_SOVEREIGN.json (env AIPROD_CONFIG ou défaut).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from .schema.schemas import Context, PipelineRequest, PipelineResponse
from .checkpoint.manager import CheckpointManager
from .checkpoint.recovery import RecoveryManager, RecoveryAction

logger = logging.getLogger(__name__)

# ---------- Config loader ----------

_DEFAULT_CONFIG_PATH = "config/AIPROD_V34_SOVEREIGN.json"


def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge la configuration pipeline.
    
    Priority:
        1. config_path argument
        2. AIPROD_CONFIG env var
        3. config/AIPROD_V34_SOVEREIGN.json (défaut souverain)
    """
    path = config_path or os.environ.get("AIPROD_CONFIG", _DEFAULT_CONFIG_PATH)
    config_file = Path(path)
    
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info("Pipeline config loaded: %s (v%s)", path, cfg.get("version", "?"))
        return cfg
    
    logger.warning("Config file not found: %s — using defaults", path)
    return {
        "version": "3.4",
        "sovereignty": {"score": "9/10", "cloudDependencies": 0},
        "state": {"blocks": {}},
    }


class Orchestrator:
    """
    Production pipeline orchestrator with checkpoint-based resilience.
    
    Executes jobs through an 11-state machine:
    INIT → ANALYSIS → CREATIVE_DIRECTION → VISUAL_TRANSLATION → 
    FINANCIAL_OPTIMIZATION → RENDER_EXECUTION → QA_TECHNICAL → 
    QA_SEMANTIC → FINALIZE
    
    With fast-track path: INIT → FAST_TRACK → FINANCIAL_OPTIMIZATION
    And error handling: * → ERROR → INIT (with retry)
    """
    
    def __init__(
        self, 
        adapters: Dict[str, Any],
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_retries: int = 3,
        config_path: Optional[str] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            adapters: Dictionary of adapter instances by name
            checkpoint_manager: CheckpointManager (created if None)
            max_retries: Maximum retry attempts per state
            config_path: Path to pipeline config JSON (default: V34 sovereign)
        """
        self.adapters = adapters
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.recovery_manager = RecoveryManager(self.checkpoint_manager, max_retries)
        self.max_retries = max_retries
        
        # Charger la config souveraine
        self.pipeline_config = load_pipeline_config(config_path)
        logger.info(
            "Orchestrator initialized (config v%s, sovereignty=%s)",
            self.pipeline_config.get("version", "?"),
            self.pipeline_config.get("sovereignty", {}).get("score", "?"),
        )
        
        # State transition map
        self.state_handlers: Dict[str, Callable] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all state handlers."""
        # Import handlers module
        from . import handlers
        
        self.state_handlers = {
            "INIT": handlers.handle_init,
            "ANALYSIS": handlers.handle_analysis,
            "CREATIVE_DIRECTION": handlers.handle_creative_direction,
            "VISUAL_TRANSLATION": handlers.handle_visual_translation,
            "FAST_TRACK": handlers.handle_fast_track,
            "FINANCIAL_OPTIMIZATION": handlers.handle_financial_optimization,
            "RENDER_EXECUTION": handlers.handle_render_execution,
            "QA_TECHNICAL": handlers.handle_qa_technical,
            "QA_SEMANTIC": handlers.handle_qa_semantic,
            "FINALIZE": handlers.handle_finalize,
            "ERROR": handlers.handle_error,
        }
    
    async def execute(self, request: PipelineRequest) -> PipelineResponse:
        """
        Execute production pipeline with checkpoint/resume.
        
        Args:
            request: Pipeline execution request
            
        Returns:
            Pipeline execution response with results
        """
        # Initialize context
        ctx = self._init_context(request)
        state = "INIT"
        attempt = 0
        checkpoints_created = 0
        errors = []
        
        # Main execution loop
        while state != "FINALIZE" and state != "ERROR":
            try:
                # SAVE CHECKPOINT BEFORE STATE EXECUTION
                checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                    job_id=ctx["request_id"],
                    state=state,
                    context=ctx
                )
                checkpoints_created += 1
                
                # Get state handler
                handler = self.state_handlers.get(state)
                if not handler:
                    raise ValueError(f"Unknown state: {state}")
                
                # Execute state handler
                next_state, ctx = await handler(ctx, self.adapters)
                
                # Mark checkpoint as successful
                await self.checkpoint_manager.mark_successful(checkpoint_id)
                
                # Transition to next state
                state = next_state
                attempt = 0  # Reset attempt counter on success
                
            except Exception as e:
                attempt += 1
                errors.append(f"{state}: {str(e)}")
                
                # Handle failure with recovery manager
                action, restored_ctx = await self.recovery_manager.handle_failure(
                    job_id=ctx["request_id"],
                    state=state,
                    error=e,
                    attempt=attempt
                )
                
                if action == RecoveryAction.RETRY and restored_ctx:
                    # Restore context and retry
                    ctx = restored_ctx
                    # state remains same, will retry
                    
                elif action == RecoveryAction.ROLLBACK:
                    # No checkpoint, start over
                    state = "INIT"
                    attempt = 0
                    
                else:
                    # ERROR or SKIP
                    state = "ERROR"
                    ctx["error"] = str(e)
        
        # Build response
        execution_time = time.time() - ctx["memory"]["start_time"]
        
        response: PipelineResponse = {
            "job_id": ctx["request_id"],
            "status": "completed" if state == "FINALIZE" else "failed",
            "delivery_manifest": ctx["memory"].get("delivery_manifest", {}),
            "cost": ctx["memory"].get("cost_estimation", {}).get("total_estimated", 0),
            "quality_score": ctx["memory"].get("quality_score", 0),
            "execution_time_sec": execution_time,
            "checkpoints_created": checkpoints_created,
            "errors": errors
        }
        
        return response
    
    def _init_context(self, request: PipelineRequest) -> Context:
        """
        Initialize execution context from request.
        
        Args:
            request: Pipeline request
            
        Returns:
            Initialized context
        """
        ctx: Context = {
            "request_id": request["request_id"],
            "state": "INIT",
            "memory": {
                "start_time": time.time(),
                "request": request,
                "prompt": request.get("prompt", ""),
                "duration_sec": request.get("duration_sec", 60),
                "budget": request.get("budget", 1.0),
                "complexity": request.get("complexity", 0.5),
                "preferences": request.get("preferences", {}),
                "fallback_enabled": request.get("fallback_enabled", True)
            },
            "config": {
                "max_retries": self.max_retries,
                "checkpoint_enabled": True,
                "pipeline": self.pipeline_config,
                "version": self.pipeline_config.get("version", "3.4"),
                "backend": "aiprod_sovereign",
            }
        }
        
        return ctx
    
    async def resume_job(self, job_id: str, checkpoint_id: Optional[str] = None) -> PipelineResponse:
        """
        Resume a failed job from checkpoint.
        
        Args:
            job_id: Job identifier to resume
            checkpoint_id: Specific checkpoint to resume from (or latest if None)
            
        Returns:
            Pipeline execution response
        """
        # Find checkpoint
        if not checkpoint_id:
            checkpoint_id = await self.checkpoint_manager.get_latest_checkpoint(job_id)
        
        if not checkpoint_id:
            raise ValueError(f"No checkpoint found for job {job_id}")
        
        # Restore context
        ctx = await self.recovery_manager.resume_from_checkpoint(job_id, checkpoint_id)
        
        if not ctx:
            raise ValueError(f"Failed to restore checkpoint {checkpoint_id}")
        
        # Resume execution from restored state
        current_state = ctx.get("state", "INIT")
        
        # Continue execution loop
        state = current_state
        attempt = 0
        checkpoints_created = 0
        errors = []
        
        while state != "FINALIZE" and state != "ERROR":
            try:
                checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                    job_id=ctx["request_id"],
                    state=state,
                    context=ctx
                )
                checkpoints_created += 1
                
                handler = self.state_handlers.get(state)
                if not handler:
                    raise ValueError(f"Unknown state: {state}")
                
                next_state, ctx = await handler(ctx, self.adapters)
                
                await self.checkpoint_manager.mark_successful(checkpoint_id)
                
                state = next_state
                attempt = 0
                
            except Exception as e:
                attempt += 1
                errors.append(f"{state}: {str(e)}")
                
                action, restored_ctx = await self.recovery_manager.handle_failure(
                    job_id=ctx["request_id"],
                    state=state,
                    error=e,
                    attempt=attempt
                )
                
                if action == RecoveryAction.RETRY and restored_ctx:
                    ctx = restored_ctx
                elif action == RecoveryAction.ROLLBACK:
                    state = "INIT"
                    attempt = 0
                else:
                    state = "ERROR"
                    ctx["error"] = str(e)
        
        execution_time = time.time() - ctx["memory"]["start_time"]
        
        response: PipelineResponse = {
            "job_id": ctx["request_id"],
            "status": "completed" if state == "FINALIZE" else "failed",
            "delivery_manifest": ctx["memory"].get("delivery_manifest", {}),
            "cost": ctx["memory"].get("cost_estimation", {}).get("total_estimated", 0),
            "quality_score": ctx["memory"].get("quality_score", 0),
            "execution_time_sec": execution_time,
            "checkpoints_created": checkpoints_created,
            "errors": errors
        }
        
        return response

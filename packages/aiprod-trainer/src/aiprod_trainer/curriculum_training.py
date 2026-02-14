# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Curriculum Learning for AIPROD Video Generation

Implements multi-phase curriculum training:
- Phase 1: Low resolution (256×256) + short duration (8 frames) + small batch
- Phase 2: High resolution (512×512) + medium duration (16 frames) + larger batch  
- Phase 3: Full resolution (1024×1024) + long duration (49+ frames) + max batch

Key features:
- Automatic phase transitions based on step count or validation metrics
- Per-phase learning rate schedules
- Progressive batch size and resolution increases
- Memory-aware scheduling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class CurriculumPhase(str, Enum):
    """Curriculum training phases."""
    PHASE_1_LOW_RES = "phase_1_low_res"
    PHASE_2_HIGH_RES = "phase_2_high_res"
    PHASE_3_FULL_RES = "phase_3_full_res"


@dataclass
class PhaseDuration:
    """Duration configuration for a curriculum phase."""
    min_frames: int = 4
    max_frames: int = 16
    target_frames: int = 8


@dataclass
class PhaseResolution:
    """Resolution configuration for a curriculum phase."""
    height: int = 256
    width: int = 256
    latent_height: int = 32
    latent_width: int = 32


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: CurriculumPhase
    
    # Duration settings
    duration: PhaseDuration = field(default_factory=PhaseDuration)
    
    # Resolution settings
    resolution: PhaseResolution = field(default_factory=PhaseResolution)
    
    # Training parameters
    batch_size: int = 1
    learning_rate: float = 1e-4
    learning_rate_multiplier: float = 1.0
    
    # Schedule (one of: "step" or "epoch")
    schedule_by: Literal["step", "epoch"] = "step"
    
    # Duration in steps or epochs
    duration_value: int = 10000
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Warmup (in steps or epochs, depending on schedule_by)
    warmup_steps: int = 500
    
    # Optional validation config
    validation_interval_multiplier: float = 1.0
    checkpoint_interval_multiplier: float = 1.0


@dataclass
class CurriculumConfig:
    """Configuration for curriculum training."""
    
    # Enable curriculum training
    enabled: bool = True
    
    # Phases (order matters!)
    phases: list[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            name=CurriculumPhase.PHASE_1_LOW_RES,
            duration=PhaseDuration(min_frames=4, max_frames=8, target_frames=8),
            resolution=PhaseResolution(height=256, width=256, latent_height=32, latent_width=32),
            batch_size=16,
            learning_rate=1e-4,
            learning_rate_multiplier=1.0,
            duration_value=50000,
            warmup_steps=1000,
            gradient_accumulation_steps=1,
        ),
        PhaseConfig(
            name=CurriculumPhase.PHASE_2_HIGH_RES,
            duration=PhaseDuration(min_frames=8, max_frames=16, target_frames=16),
            resolution=PhaseResolution(height=512, width=512, latent_height=64, latent_width=64),
            batch_size=8,
            learning_rate=5e-5,
            learning_rate_multiplier=0.5,
            duration_value=100000,
            warmup_steps=500,
            gradient_accumulation_steps=2,
        ),
        PhaseConfig(
            name=CurriculumPhase.PHASE_3_FULL_RES,
            duration=PhaseDuration(min_frames=16, max_frames=49, target_frames=49),
            resolution=PhaseResolution(height=1024, width=1024, latent_height=128, latent_width=128),
            batch_size=4,
            learning_rate=2e-5,
            learning_rate_multiplier=0.2,
            duration_value=150000,
            warmup_steps=500,
            gradient_accumulation_steps=4,
        ),
    ])
    
    # Transition strategy
    transition_on: Literal["step", "epoch", "loss_plateau"] = "step"
    
    # Loss improvement threshold for plateau detection
    plateau_threshold: float = 0.001  # 0.1% improvement required
    plateau_window: int = 1000  # Check over this many steps


class CurriculumScheduler:
    """Manages curriculum phase transitions."""
    
    def __init__(self, curriculum_config: CurriculumConfig):
        self.config = curriculum_config
        self.current_phase_idx = 0
        self.phase_step_counter = 0
        self.phase_epoch_counter = 0
        self.loss_history = []
        
    @property
    def current_phase(self) -> PhaseConfig:
        """Get current phase config."""
        return self.config.phases[self.current_phase_idx]
    
    @property
    def is_single_phase(self) -> bool:
        """Check if only one phase is configured."""
        return len(self.config.phases) <= 1
    
    def step(self) -> bool:
        """
        Increment step counter and check if phase transition is needed.
        
        Returns:
            True if phase transitioned, False otherwise
        """
        if not self.config.enabled or self.is_single_phase:
            return False
        
        self.phase_step_counter += 1
        
        # Check if should transition
        if self._should_transition():
            return self._transition_phase()
        
        return False
    
    def epoch(self) -> bool:
        """
        Increment epoch counter and check if phase transition is needed.
        
        Returns:
            True if phase transitioned, False otherwise
        """
        if not self.config.enabled or self.is_single_phase:
            return False
        
        self.phase_epoch_counter += 1
        
        # Check if should transition
        if self._should_transition():
            return self._transition_phase()
        
        return False
    
    def record_loss(self, loss: float) -> None:
        """Record loss for plateau detection."""
        self.loss_history.append(loss)
    
    def _should_transition(self) -> bool:
        """Check if should transition to next phase."""
        if self.current_phase_idx >= len(self.config.phases) - 1:
            # Already at last phase
            return False
        
        phase = self.current_phase
        
        if self.config.transition_on == "step":
            return self.phase_step_counter >= phase.duration_value
        elif self.config.transition_on == "epoch":
            return self.phase_epoch_counter >= phase.duration_value
        elif self.config.transition_on == "loss_plateau":
            return self._check_loss_plateau()
        
        return False
    
    def _check_loss_plateau(self) -> bool:
        """Check if loss has plateaued (no significant improvement)."""
        if len(self.loss_history) < self.config.plateau_window:
            return False
        
        recent_loss = self.loss_history[-self.config.plateau_window:]
        initial_loss = recent_loss[0]
        final_loss = recent_loss[-1]
        
        # Check if improvement is below threshold
        improvement = (initial_loss - final_loss) / (initial_loss + 1e-8)
        return improvement < self.config.plateau_threshold
    
    def _transition_phase(self) -> bool:
        """Transition to next phase."""
        if self.current_phase_idx >= len(self.config.phases) - 1:
            return False
        
        self.current_phase_idx += 1
        self.phase_step_counter = 0
        self.phase_epoch_counter = 0
        self.loss_history = []
        
        phase = self.current_phase
        logger.info(
            f"🔄 Transitioned to {phase.name.value} "
            f"({phase.resolution.height}×{phase.resolution.width}, "
            f"{phase.duration.target_frames} frames)"
        )
        
        return True
    
    def get_phase_info(self) -> dict:
        """Get information about current phase."""
        phase = self.current_phase
        return {
            "phase": phase.name.value,
            "phase_index": self.current_phase_idx + 1,
            "total_phases": len(self.config.phases),
            "resolution": f"{phase.resolution.height}×{phase.resolution.width}",
            "duration_frames": phase.duration.target_frames,
            "batch_size": phase.batch_size,
            "learning_rate": phase.learning_rate,
            "step_in_phase": self.phase_step_counter,
            "epoch_in_phase": self.phase_epoch_counter,
        }


class CurriculumAdapterConfig:
    """
    Adapter to integrate curriculum training with AIPRODTrainerConfig.
    
    This adapter bridges the gap between curriculum phases and the main trainer config.
    """
    
    def __init__(self, base_config, curriculum_config: CurriculumConfig):
        self.base_config = base_config
        self.curriculum_config = curriculum_config
        self.scheduler = CurriculumScheduler(curriculum_config)
    
    def get_current_phase_config(self) -> PhaseConfig:
        """Get current curriculum phase config."""
        return self.scheduler.current_phase
    
    def apply_phase_to_trainer_config(self) -> None:
        """Apply current phase settings to the base trainer config."""
        if not self.curriculum_config.enabled:
            return
        
        phase = self.scheduler.current_phase
        
        # Update batch size
        if hasattr(self.base_config.optimization, "batch_size"):
            self.base_config.optimization.batch_size = phase.batch_size
        
        # Update learning rate
        if hasattr(self.base_config.optimization, "learning_rate"):
            self.base_config.optimization.learning_rate = (
                phase.learning_rate * phase.learning_rate_multiplier
            )
        
        # Update gradient accumulation
        if hasattr(self.base_config.optimization, "gradient_accumulation_steps"):
            self.base_config.optimization.gradient_accumulation_steps = phase.gradient_accumulation_steps
        
        # Update validation interval
        if hasattr(self.base_config.validation, "interval"):
            base_interval = self.base_config.validation.interval
            if base_interval is not None:
                self.base_config.validation.interval = int(
                    base_interval * phase.validation_interval_multiplier
                )
        
        # Update checkpoint interval
        if hasattr(self.base_config.checkpoints, "interval"):
            base_interval = self.base_config.checkpoints.interval
            if base_interval is not None:
                self.base_config.checkpoints.interval = int(
                    base_interval * phase.checkpoint_interval_multiplier
                )
    
    def should_transition(self) -> bool:
        """Check if should transition to next phase."""
        is_advancing = (
            self.scheduler.step()
            if self.curriculum_config.transition_on == "step"
            else self.scheduler.epoch()
        )
        
        if is_advancing:
            self.apply_phase_to_trainer_config()
            phase_info = self.scheduler.get_phase_info()
            logger.info(f"📊 Phase transition details: {phase_info}")
        
        return is_advancing


__all__ = [
    "CurriculumPhase",
    "PhaseDuration",
    "PhaseResolution",
    "PhaseConfig",
    "CurriculumConfig",
    "CurriculumScheduler",
    "CurriculumAdapterConfig",
]

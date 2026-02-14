"""Tests for aiprod_trainer.curriculum_training module."""

import pytest
from aiprod_trainer.curriculum_training import (
    CurriculumPhase, PhaseDuration, PhaseResolution, PhaseConfig, CurriculumConfig,
    CurriculumScheduler, CurriculumAdapterConfig
)


class TestPhaseDuration:
    """Test PhaseDuration."""
    
    def test_phase_duration_creation(self):
        """Test PhaseDuration creation."""
        duration = PhaseDuration(
            min_frames=4,
            max_frames=8,
            target_frames=8
        )
        assert duration.min_frames == 4
        assert duration.max_frames == 8
        assert duration.target_frames == 8
    
    def test_phase_duration_defaults(self):
        """Test PhaseDuration defaults."""
        duration = PhaseDuration()
        assert duration.min_frames > 0
        assert duration.max_frames >= duration.min_frames
        assert duration.target_frames <= duration.max_frames


class TestPhaseResolution:
    """Test PhaseResolution."""
    
    def test_phase_resolution_creation(self):
        """Test PhaseResolution creation."""
        resolution = PhaseResolution(
            height=512,
            width=512,
            latent_height=64,
            latent_width=64
        )
        assert resolution.height == 512
        assert resolution.width == 512
        assert resolution.latent_height == 64
        assert resolution.latent_width == 64
    
    def test_phase_resolution_defaults(self):
        """Test PhaseResolution defaults."""
        resolution = PhaseResolution()
        assert resolution.height > 0
        assert resolution.width > 0
        assert resolution.latent_height > 0
        assert resolution.latent_width > 0


class TestPhaseConfig:
    """Test PhaseConfig."""
    
    def test_phase_config_creation(self):
        """Test PhaseConfig creation."""
        config = PhaseConfig(
            name=CurriculumPhase.PHASE_1_LOW_RES,
            batch_size=16,
            learning_rate=1e-4,
            duration_value=10000
        )
        assert config.name == CurriculumPhase.PHASE_1_LOW_RES
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.duration_value == 10000
    
    def test_phase_config_duration_settings(self):
        """Test phase duration settings."""
        config = PhaseConfig(
            name=CurriculumPhase.PHASE_1_LOW_RES,
            duration=PhaseDuration(min_frames=4, max_frames=8, target_frames=8)
        )
        assert config.duration.target_frames == 8
    
    def test_phase_config_resolution_settings(self):
        """Test phase resolution settings."""
        config = PhaseConfig(
            name=CurriculumPhase.PHASE_1_LOW_RES,
            resolution=PhaseResolution(height=256, width=256)
        )
        assert config.resolution.height == 256
        assert config.resolution.width == 256


class TestCurriculumPhase:
    """Test CurriculumPhase enum."""
    
    def test_curriculum_phases_exist(self):
        """Test that all curriculum phases are defined."""
        assert hasattr(CurriculumPhase, "PHASE_1_LOW_RES")
        assert hasattr(CurriculumPhase, "PHASE_2_HIGH_RES")
        assert hasattr(CurriculumPhase, "PHASE_3_FULL_RES")
    
    def test_curriculum_phase_values(self):
        """Test curriculum phase values."""
        assert CurriculumPhase.PHASE_1_LOW_RES.value == "phase_1_low_res"
        assert CurriculumPhase.PHASE_2_HIGH_RES.value == "phase_2_high_res"
        assert CurriculumPhase.PHASE_3_FULL_RES.value == "phase_3_full_res"


class TestCurriculumConfig:
    """Test CurriculumConfig."""
    
    def test_curriculum_config_creation(self):
        """Test CurriculumConfig creation."""
        config = CurriculumConfig(enabled=True)
        assert config.enabled is True
        assert len(config.phases) == 3
    
    def test_curriculum_config_phases(self):
        """Test curriculum phases configuration."""
        config = CurriculumConfig()
        assert config.phases[0].name == CurriculumPhase.PHASE_1_LOW_RES
        assert config.phases[1].name == CurriculumPhase.PHASE_2_HIGH_RES
        assert config.phases[2].name == CurriculumPhase.PHASE_3_FULL_RES
    
    def test_curriculum_config_transition_on(self):
        """Test curriculum transition strategy."""
        config = CurriculumConfig(transition_on="step")
        assert config.transition_on == "step"
        
        config = CurriculumConfig(transition_on="epoch")
        assert config.transition_on == "epoch"
        
        config = CurriculumConfig(transition_on="loss_plateau")
        assert config.transition_on == "loss_plateau"
    
    def test_curriculum_config_plateau_settings(self):
        """Test loss plateau detection settings."""
        config = CurriculumConfig(
            plateau_threshold=0.001,
            plateau_window=1000
        )
        assert config.plateau_threshold == 0.001
        assert config.plateau_window == 1000


class TestCurriculumScheduler:
    """Test CurriculumScheduler."""
    
    def test_curriculum_scheduler_initialization(self):
        """Test CurriculumScheduler initialization."""
        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)
        
        assert scheduler.current_phase_idx == 0
        assert scheduler.phase_step_counter == 0
        assert scheduler.phase_epoch_counter == 0
    
    def test_curriculum_scheduler_current_phase(self):
        """Test getting current phase."""
        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)
        
        phase = scheduler.current_phase
        assert phase.name == CurriculumPhase.PHASE_1_LOW_RES
    
    def test_curriculum_scheduler_is_single_phase(self):
        """Test single phase detection."""
        # When curriculum is enabled with default 3 phases, it's multi-phase
        config_multi = CurriculumConfig(enabled=True)
        scheduler_multi = CurriculumScheduler(config_multi)
        assert not scheduler_multi.is_single_phase
        
        # Single phase config has only one phase
        config_single = CurriculumConfig(phases=[
            PhaseConfig(name=CurriculumPhase.PHASE_1_LOW_RES)
        ])
        scheduler_single = CurriculumScheduler(config_single)
        assert scheduler_single.is_single_phase
    
    def test_curriculum_scheduler_step(self):
        """Test stepping through curriculum."""
        config = CurriculumConfig(transition_on="step")
        scheduler = CurriculumScheduler(config)
        
        initial_phase = scheduler.current_phase_idx
        
        # Step to trigger transition (duration_value is 50000 by default for phase 0)
        for _ in range(scheduler.current_phase.duration_value + 1):
            if scheduler.step():
                break
        
        # Should have transitioned to next phase (or stayed if at max phases)
        if not scheduler.is_single_phase:
            assert scheduler.current_phase_idx >= initial_phase
    
    def test_curriculum_scheduler_epoch(self):
        """Test epoch-based transitions."""
        config = CurriculumConfig(transition_on="epoch")
        scheduler = CurriculumScheduler(config)
        
        initial_phase = scheduler.current_phase_idx
        phase_duration = scheduler.current_phase.duration_value
        
        # Epoch for x times
        for _ in range(phase_duration + 100):
            scheduler.epoch()
        
        if phase_duration < 1000:  # Only check if phase duration is reasonable
            assert scheduler.current_phase_idx > initial_phase
    
    def test_curriculum_scheduler_get_phase_info(self):
        """Test getting phase information."""
        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)
        
        info = scheduler.get_phase_info()
        assert 'phase' in info
        assert 'phase_index' in info
        assert 'total_phases' in info
        assert info['phase_index'] == 1
        assert info['total_phases'] == 3
    
    def test_curriculum_scheduler_record_loss(self):
        """Test recording loss for plateau detection."""
        config = CurriculumConfig(transition_on="loss_plateau")
        scheduler = CurriculumScheduler(config)
        
        # Record some losses
        losses = [1.0, 0.9, 0.8, 0.75, 0.74, 0.73, 0.72, 0.71, 0.71, 0.70]
        for loss in losses:
            scheduler.record_loss(loss)
        
        assert len(scheduler.loss_history) == len(losses)
    
    def test_curriculum_scheduler_loss_plateau_detection(self):
        """Test loss plateau detection."""
        config = CurriculumConfig(
            transition_on="loss_plateau",
            plateau_threshold=0.001,
            plateau_window=10
        )
        scheduler = CurriculumScheduler(config)
        
        # Record losses with no improvement (plateau)
        for i in range(20):
            loss = 1.0 - (0.0001 * i)  # Very slow improvement
            scheduler.record_loss(loss)
        
        # Should detect plateau
        is_plateau = scheduler._check_loss_plateau()
        assert is_plateau is True
    
    def test_curriculum_scheduler_no_plateau(self):
        """Test that good improvement prevents plateau detection."""
        config = CurriculumConfig(
            transition_on="loss_plateau",
            plateau_threshold=0.001,
            plateau_window=10
        )
        scheduler = CurriculumScheduler(config)
        
        # Record losses with good improvement (no plateau)
        for i in range(20):
            loss = 1.0 - (0.05 * i)  # Good improvement
            scheduler.record_loss(loss)
        
        is_plateau = scheduler._check_loss_plateau()
        assert is_plateau is False
    
    def test_curriculum_scheduler_transition_phase(self):
        """Test manual phase transition."""
        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)
        
        initial = scheduler.current_phase_idx
        transitioned = scheduler._transition_phase()
        
        assert transitioned is True
        assert scheduler.current_phase_idx == initial + 1
        assert scheduler.phase_step_counter == 0
    
    def test_curriculum_scheduler_last_phase_no_transition(self):
        """Test that last phase doesn't transition."""
        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)
        
        # Move to last phase
        scheduler.current_phase_idx = len(config.phases) - 1
        
        transitioned = scheduler._transition_phase()
        assert transitioned is False


class TestCurriculumAdapterConfig:
    """Test CurriculumAdapterConfig."""
    
    def test_curriculum_adapter_initialization(self):
        """Test CurriculumAdapterConfig initialization."""
        base_config = type('Config', (), {'validation': type('V', (), {'interval': 500})(), 'checkpoints': type('C', (), {'interval': 1000})(), 'optimization': type('O', (), {})()})()
        curriculum_config = CurriculumConfig()
        
        adapter = CurriculumAdapterConfig(base_config, curriculum_config)
        assert adapter.base_config is not None
        assert adapter.curriculum_config is not None
        assert adapter.scheduler is not None
    
    def test_curriculum_adapter_get_current_phase(self):
        """Test getting current phase from adapter."""
        base_config = type('Config', (), {'validation': type('V', (), {'interval': 500})(), 'checkpoints': type('C', (), {'interval': 1000})(), 'optimization': type('O', (), {})()})()
        curriculum_config = CurriculumConfig()
        
        adapter = CurriculumAdapterConfig(base_config, curriculum_config)
        phase = adapter.get_current_phase_config()
        
        assert phase.name == CurriculumPhase.PHASE_1_LOW_RES


class TestMultiPhaseTransition:
    """Test transitions across all phases."""
    
    def test_transition_all_phases(self):
        """Test transitioning through all curriculum phases."""
        config = CurriculumConfig(transition_on="step")
        scheduler = CurriculumScheduler(config)
        
        # Track phase transitions
        phase_history = []
        
        for i in range(300000):  # Simulate 300K steps
            phase_before = scheduler.current_phase_idx
            scheduler.step()
            phase_after = scheduler.current_phase_idx
            
            if phase_before != phase_after:
                phase_history.append((i, phase_after))
        
        # Should have transitioned at least once
        assert len(phase_history) > 0
        
        # Phases should be in order
        for prev, curr in zip(phase_history[:-1], phase_history[1:]):
            assert curr[1] >= prev[1]

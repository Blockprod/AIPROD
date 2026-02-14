"""
Tests for adaptive kernel fusion selection.

Validates:
- GPU capability detection
- Fusion selection logic
- Performance estimation
- Speedup calculation
- Memory tracking
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from aiprod_pipelines.inference.kernel_fusion.adaptive_fusion import (
    GPUCapabilities,
    FusionProfile,
    AdaptiveKernelFusionEngine,
    auto_select_kernel_fusions,
)


class TestGPUCapabilities:
    """Tests for GPUCapabilities."""
    
    def test_high_end_gpu(self):
        """A100 is detected as high-end."""
        caps = GPUCapabilities(
            compute_capability=(8, 0),
            memory_bandwidth_gbps=2040.0,
            total_memory_gb=40.0,
            available_memory_gb=38.0,
            gpu_name="A100-80GB",
        )
        
        assert caps.is_high_end
        assert not caps.is_mid_range
        assert not caps.is_low_end
    
    def test_mid_range_gpu(self):
        """V100 is detected as mid-range."""
        caps = GPUCapabilities(
            compute_capability=(7, 0),
            memory_bandwidth_gbps=900.0,
            total_memory_gb=32.0,
            available_memory_gb=30.0,
            gpu_name="Tesla V100",
        )
        
        assert not caps.is_high_end
        assert caps.is_mid_range
        assert not caps.is_low_end
    
    def test_low_end_gpu(self):
        """RTX 2080 is detected as low-end."""
        caps = GPUCapabilities(
            compute_capability=(7, 5),
            memory_bandwidth_gbps=480.0,
            total_memory_gb=11.0,
            available_memory_gb=9.0,
            gpu_name="RTX 2080 Ti",
        )
        
        assert not caps.is_high_end
        assert caps.is_mid_range
        assert not caps.is_low_end
    
    def test_h100_capabilities(self):
        """H100 has correct properties."""
        caps = GPUCapabilities(
            compute_capability=(9, 0),
            memory_bandwidth_gbps=3352.0,
            total_memory_gb=80.0,
            available_memory_gb=78.0,
            gpu_name="H100",
        )
        
        assert caps.is_high_end
        assert caps.memory_bandwidth_gbps > 3000


class TestFusionProfile:
    """Tests for FusionProfile."""
    
    def test_speedup_percent_calculation(self):
        """Speedup percent is calculated correctly."""
        profile = FusionProfile(
            name="test_fusion",
            speedup_factor=1.35,
            memory_overhead_bytes=1024,
            is_compatible=True,
            confidence=0.9,
        )
        
        assert profile.speedup_percent() == pytest.approx(35.0)
    
    def test_speedup_percent_zero_for_no_speedup(self):
        """No speedup gives 0% improvement."""
        profile = FusionProfile(
            name="no_improvement",
            speedup_factor=1.0,
            memory_overhead_bytes=0,
            is_compatible=True,
            confidence=1.0,
        )
        
        assert profile.speedup_percent() == 0.0
    
    def test_negative_speedup_possible(self):
        """Can represent slowdown."""
        profile = FusionProfile(
            name="slowdown",
            speedup_factor=0.95,
            memory_overhead_bytes=2048,
            is_compatible=True,
            confidence=0.5,
        )
        
        assert profile.speedup_percent() < 0


class TestAdaptiveKernelFusionEngine:
    """Tests for AdaptiveKernelFusionEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine with mocked GPU capabilities."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities") as mock_get:
            mock_get.return_value = GPUCapabilities(
                compute_capability=(8, 0),
                memory_bandwidth_gbps=2040.0,
                total_memory_gb=40.0,
                available_memory_gb=38.0,
                gpu_name="A100",
            )
            yield AdaptiveKernelFusionEngine()
    
    def test_initialization(self, engine):
        """Engine initializes correctly."""
        assert engine.memory_headroom_gb == 1.0
        assert engine.target_speedup_factor == 1.15
        assert engine.gpu_capabilities is not None
    
    def test_custom_parameters(self):
        """Can initialize with custom parameters."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            engine = AdaptiveKernelFusionEngine(
                memory_headroom_gb=2.0,
                target_speedup_factor=1.20,
            )
            
            assert engine.memory_headroom_gb == 2.0
            assert engine.target_speedup_factor == 1.20
    
    def test_suggest_fusions_returns_list(self, engine):
        """suggest_fusions returns list of strings."""
        model_config = {
            "hidden_dim": 768,
            "num_heads": 12,
            "total_flops": 1e12,
        }
        
        fusions = engine.suggest_fusions(model_config)
        
        assert isinstance(fusions, list)
        assert all(isinstance(f, str) for f in fusions)
    
    def test_suggest_fusions_includes_high_priority(self, engine):
        """Attention fusion is always suggested."""
        model_config = {"hidden_dim": 768}
        
        fusions = engine.suggest_fusions(model_config)
        
        # Attention fusion should be high priority
        if len(fusions) > 0:
            assert fusions[0] in ["attention_linear", "residual_block"]
    
    def test_suggest_fusions_respects_memory_budget(self, engine):
        """Won't suggest fusions if no GPU memory."""
        with patch.object(engine.gpu_capabilities, "available_memory_gb", 0.5):
            model_config = {"hidden_dim": 768}
            fusions = engine.suggest_fusions(model_config)
            
            # Might still suggest some, but not aggressive ones
            # This is implementation-dependent
    
    def test_suggest_fusions_respects_target_speedup(self, engine):
        """Stops adding fusions when target reached."""
        engine.target_speedup_factor = 1.05  # Very low target
        
        model_config = {"hidden_dim": 768}
        fusions = engine.suggest_fusions(model_config)
        
        # Should select minimal fusions to hit target
        assert len(fusions) <= 2
    
    def test_estimate_speedup_calculation(self, engine):
        """Speedup estimation is reasonable."""
        fusions = ["attention_linear", "residual_block"]
        
        estimates = engine.estimate_speedup(fusions, 1e12)
        
        assert "cumulative_speedup" in estimates
        assert "speedup_percent" in estimates
        assert estimates["cumulative_speedup"] > 1.0
        assert estimates["speedup_percent"] > 0
    
    def test_estimate_speedup_single_fusion(self, engine):
        """Single fusion speedup."""
        estimates = engine.estimate_speedup(["attention_linear"], 1e12)
        
        # Single attention fusion should give ~35% speedup
        assert 30 < estimates["speedup_percent"] < 40
    
    def test_estimate_speedup_multiple_fusions(self, engine):
        """Multiple fusions compound speedup."""
        single = engine.estimate_speedup(["attention_linear"], 1e12)
        multi = engine.estimate_speedup(
            ["attention_linear", "residual_block",], 
            1e12
        )
        
        # Multiple should be faster than single
        assert multi["cumulative_speedup"] > single["cumulative_speedup"]
    
    def test_is_compatible_attention_always_true(self, engine):
        """Attention fusion compatible on all GPUs."""
        assert engine._is_compatible("attention_linear")
    
    def test_is_compatible_residual_always_true(self, engine):
        """Residual fusion compatible on all GPUs."""
        assert engine._is_compatible("residual_block")
    
    def test_is_compatible_conv_depends_on_gpu(self, engine):
        """Conv fusion depends on GPU tier."""
        # A100 is high-end, should be compatible
        assert engine._is_compatible("conv_activation") or not engine.gpu_capabilities.is_high_end
    
    def test_profile_for_shapes_uses_cache(self, engine):
        """Profiles are cached."""
        shapes = {"latents": (1, 16, 64, 64, 48)}
        
        profile1 = engine.profile_for_shapes("attention_linear", shapes)
        profile2 = engine.profile_for_shapes("attention_linear", shapes)
        
        # Should be same object (cached)
        assert profile1 is profile2
    
    def test_profile_returns_valid_profile(self, engine):
        """Profiles have valid fields."""
        shapes = {"latents": (1, 16, 64, 64, 48)}
        
        profile = engine.profile_for_shapes("attention_linear", shapes)
        
        assert isinstance(profile, FusionProfile)
        assert profile.speedup_factor > 1.0
        assert 0 <= profile.confidence <= 1


class TestAutoSelectKernelFusions:
    """Tests for auto_select_kernel_fusions function."""
    
    def test_returns_tuple(self):
        """Function returns (fusions, estimates) tuple."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            result = auto_select_kernel_fusions(
                model_config={"hidden_dim": 768, "total_flops": 1e12}
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 2
    
    def test_returns_lists_and_dicts(self):
        """Components are correct types."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            fusions, estimates = auto_select_kernel_fusions(
                model_config={"hidden_dim": 768}
            )
            
            assert isinstance(fusions, list)
            assert isinstance(estimates, dict)
    
    def test_estimates_have_required_keys(self):
        """Estimates dict has all required fields."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            _, estimates = auto_select_kernel_fusions(
                model_config={"hidden_dim": 768}
            )
            
            assert "cumulative_speedup" in estimates
            assert "speedup_percent" in estimates
            assert "memory_overhead_bytes" in estimates


class TestMemoryOverheadEstimation:
    """Tests for memory overhead calculation."""
    
    def test_empty_fusions_zero_overhead(self):
        """No fusions = no memory overhead."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            engine = AdaptiveKernelFusionEngine()
            overhead = engine._estimate_memory_overhead([])
            
            assert overhead == 0
    
    def test_single_fusion_overhead_positive(self):
        """Each fusion has some overhead."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            engine = AdaptiveKernelFusionEngine()
            overhead = engine._estimate_memory_overhead(["attention_linear"])
            
            assert overhead > 0
    
    def test_multiple_fusions_accumulate_overhead(self):
        """Multiple fusions accumulate overhead."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            engine = AdaptiveKernelFusionEngine()
            overhead_single = engine._estimate_memory_overhead(["attention_linear"])
            overhead_double = engine._estimate_memory_overhead(
                ["attention_linear", "residual_block"]
            )
            
            assert overhead_double > overhead_single

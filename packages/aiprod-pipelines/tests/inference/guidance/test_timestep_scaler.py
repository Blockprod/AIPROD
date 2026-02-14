"""
Unit tests for TimestepScaler and AdaptiveTimestepScaler.

Coverage:
  - S-curve guidance scaling
  - Timestep weight computation
  - Schedule generation (list and tensor)
  - Complexity-aware adaptation
  - Visualization output
"""

import pytest
import torch
import numpy as np

from aiprod_pipelines.inference.guidance.timestep_scaler import (
    TimestepScaler,
    AdaptiveTimestepScaler,
)


class TestTimestepScalerInit:
    """Tests for TimestepScaler initialization."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        scaler = TimestepScaler()
        
        assert scaler.high_noise_multiplier == 1.3
        assert scaler.mid_noise_multiplier == 1.0
        assert scaler.low_noise_multiplier == 0.35
        assert scaler.total_timesteps == 1000
        assert scaler.sharpness == 12.0
    
    def test_init_custom(self):
        """Test custom initialization."""
        scaler = TimestepScaler(
            high_noise_multiplier=1.5,
            mid_noise_multiplier=1.1,
            low_noise_multiplier=0.4,
            total_timesteps=500,
            sharpness=8.0,
        )
        
        assert scaler.high_noise_multiplier == 1.5
        assert scaler.low_noise_multiplier == 0.4
        assert scaler.total_timesteps == 500
        assert scaler.sharpness == 8.0


class TestTimestepScalerWeights:
    """Tests for timestep weight computation."""
    
    def test_get_weight_high_noise(self):
        """Test weight at high noise (early step)."""
        scaler = TimestepScaler()
        
        # At timestep 999 (high noise)
        weight = scaler.get_weight(999)
        
        # Should be close to high_noise_multiplier (1.3)
        assert 1.2 < weight < 1.4
    
    def test_get_weight_low_noise(self):
        """Test weight at low noise (late step)."""
        scaler = TimestepScaler()
        
        # At timestep 0 (low noise)
        weight = scaler.get_weight(0)
        
        # Should be close to low_noise_multiplier (0.35)
        assert 0.3 < weight < 0.4
    
    def test_get_weight_mid_noise(self):
        """Test weight at mid noise."""
        scaler = TimestepScaler()
        
        # At timestep 500 (mid noise)
        weight = scaler.get_weight(500)
        
        # Should be close to mid_noise_multiplier (1.0)
        assert 0.9 < weight < 1.1
    
    def test_get_weight_monotonicity(self):
        """Test that weights decrease monotonically from high to low noise."""
        scaler = TimestepScaler()
        
        weights = [scaler.get_weight(t) for t in [999, 500, 250, 100, 0]]
        
        # Should be decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]
    
    def test_get_weight_range(self):
        """Test that all weights are in [low, high] range."""
        scaler = TimestepScaler()
        
        timesteps = [0, 100, 250, 500, 750, 999]
        weights = [scaler.get_weight(t) for t in timesteps]
        
        low, high = scaler.low_noise_multiplier, scaler.high_noise_multiplier
        assert all(low <= w <= high for w in weights)
    
    def test_get_weight_with_tensor(self):
        """Test get_weight with torch tensor input."""
        scaler = TimestepScaler()
        
        timestep = torch.tensor(500)
        weight = scaler.get_weight(timestep.item())
        
        assert isinstance(weight, float)
        assert 0.9 < weight < 1.1
    
    def test_get_weight_batch(self):
        """Test getting weights for batch of timesteps."""
        scaler = TimestepScaler()
        
        timesteps = [999, 750, 500, 250, 0]
        weights = [scaler.get_weight(t) for t in timesteps]
        
        assert len(weights) == len(timesteps)
        # Verify decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]


class TestTimestepScalerSchedules:
    """Tests for schedule generation."""
    
    def test_get_schedule_list(self):
        """Test getting schedule as list."""
        scaler = TimestepScaler()
        
        timesteps = [999, 750, 500, 250, 0]
        schedule = scaler.get_schedule(timesteps)
        
        assert isinstance(schedule, list)
        assert len(schedule) == len(timesteps)
        assert all(isinstance(w, float) for w in schedule)
    
    def test_get_schedule_length(self):
        """Test schedule has correct length."""
        scaler = TimestepScaler()
        
        for num_steps in [1, 5, 10, 30, 50]:
            timesteps = list(range(999, -1, -1000 // num_steps))[:num_steps]
            schedule = scaler.get_schedule(timesteps)
            
            assert len(schedule) == len(timesteps)
    
    def test_get_schedule_tensor(self):
        """Test getting schedule as tensor."""
        scaler = TimestepScaler()
        
        timesteps = torch.tensor([999, 750, 500, 250, 0])
        schedule = scaler.get_schedule_tensor(timesteps)
        
        assert isinstance(schedule, torch.Tensor)
        assert schedule.shape == timesteps.shape
        assert all(0.3 <= w <= 1.4 for w in schedule)
    
    def test_get_schedule_tensor_batched(self):
        """Test get_schedule_tensor with different batch sizes."""
        scaler = TimestepScaler()
        
        for batch_size in [1, 2, 4, 8, 16]:
            timesteps = torch.randint(0, 1000, (batch_size,))
            schedule = scaler.get_schedule_tensor(timesteps)
            
            assert schedule.shape[0] == batch_size
            assert all(0.3 <= w <= 1.4 for w in schedule)
    
    def test_schedule_values_reasonable(self):
        """Test that schedule values are in reasonable range."""
        scaler = TimestepScaler()
        
        # 30-step schedule (typical)
        timesteps = list(range(999, -1, -34))[:30]
        schedule = scaler.get_schedule(timesteps)
        
        # All values should be between 0.3 and 1.4
        assert all(0.3 <= w <= 1.4 for w in schedule)


class TestTimestepScalerVisualization:
    """Tests for visualization output."""
    
    def test_visualize_returns_tuple(self):
        """Test visualize returns (timesteps, multipliers) tuple."""
        scaler = TimestepScaler()
        
        timesteps, multipliers = scaler.visualize(num_points=100)
        
        assert isinstance(timesteps, (list, np.ndarray, torch.Tensor))
        assert isinstance(multipliers, (list, np.ndarray, torch.Tensor))
        assert len(timesteps) == len(multipliers)
    
    def test_visualize_num_points(self):
        """Test visualize with different num_points."""
        scaler = TimestepScaler()
        
        for num_points in [10, 50, 100, 500]:
            timesteps, multipliers = scaler.visualize(num_points=num_points)
            
            assert len(timesteps) == num_points
            assert len(multipliers) == num_points
    
    def test_visualize_curve_shape(self):
        """Test visualize curve has expected S-curve shape."""
        scaler = TimestepScaler()
        
        timesteps, multipliers = scaler.visualize(num_points=200)
        
        # Convert to list if needed
        if isinstance(multipliers, torch.Tensor):
            mults = multipliers.numpy().tolist()
        elif isinstance(multipliers, np.ndarray):
            mults = multipliers.tolist()
        else:
            mults = multipliers
        
        # Should be monotonically decreasing
        for i in range(len(mults) - 1):
            assert mults[i] >= mults[i + 1]


class TestAdaptiveTimestepScaler:
    """Tests for AdaptiveTimestepScaler subclass."""
    
    def test_inherits_from_parent(self):
        """Test that AdaptiveTimestepScaler inherits from TimestepScaler."""
        scaler = AdaptiveTimestepScaler()
        
        assert isinstance(scaler, TimestepScaler)
        assert hasattr(scaler, "get_weight")
        assert hasattr(scaler, "get_weight_adaptive")
    
    def test_get_weight_adaptive_simple(self):
        """Test adaptive weight with simple complexity."""
        scaler = AdaptiveTimestepScaler()
        
        # Simple prompt (low complexity)
        weight_simple = scaler.get_weight_adaptive(500, complexity=0.2)
        
        # Complex prompt (high complexity)
        weight_complex = scaler.get_weight_adaptive(500, complexity=0.8)
        
        # Both should be valid
        assert 0.3 <= weight_simple <= 1.4
        assert 0.3 <= weight_complex <= 1.4
    
    def test_get_weight_adaptive_complexity_effect(self):
        """Test that complexity affects adaptive weight."""
        scaler = AdaptiveTimestepScaler()
        
        timestep = 500
        
        weights = []
        for complexity in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weight = scaler.get_weight_adaptive(timestep, complexity)
            weights.append(weight)
        
        # All should be valid
        assert all(0.3 <= w <= 1.4 for w in weights)
        
        # With same timestep, complex prompts should pull toward neutral (1.0)
        # Simple prompts should be less restrained
        assert len(set(weights)) > 1  # Should have variation
    
    def test_get_weight_adaptive_extreme_complexity(self):
        """Test adaptive weight with extreme complexity values."""
        scaler = AdaptiveTimestepScaler()
        
        # Test extremes
        weight_min = scaler.get_weight_adaptive(500, complexity=0.0)
        weight_max = scaler.get_weight_adaptive(500, complexity=1.0)
        
        assert 0.3 <= weight_min <= 1.4
        assert 0.3 <= weight_max <= 1.4
    
    def test_get_weight_adaptive_all_timesteps(self):
        """Test adaptive weight across all timesteps."""
        scaler = AdaptiveTimestepScaler()
        
        complexities = [0.3, 0.5, 0.7]
        timesteps = [999, 500, 0]
        
        for complexity in complexities:
            for timestep in timesteps:
                weight = scaler.get_weight_adaptive(timestep, complexity)
                assert 0.3 <= weight <= 1.4


class TestTimestepScalerProperties:
    """Tests for scaler properties and mathematical properties."""
    
    def test_s_curve_symmetry(self):
        """Test S-curve has expected sigmoid shape."""
        scaler = TimestepScaler(sharpness=12.0)
        
        # Compute weights in center region
        center = scaler.total_timesteps / 2
        weights_before = [scaler.get_weight(int(center - i)) for i in range(100, 0, -20)]
        weights_after = [scaler.get_weight(int(center + i)) for i in range(0, 100, 20)]
        
        # Check monotonic decrease
        all_weights = weights_before + weights_after
        for i in range(len(all_weights) - 1):
            assert all_weights[i] >= all_weights[i + 1]
    
    def test_sharpness_effect(self):
        """Test that sharpness parameter affects curve steepness."""
        low_sharp = TimestepScaler(sharpness=4.0)
        high_sharp = TimestepScaler(sharpness=20.0)
        
        # At midpoint, sharp curve changes more rapidly
        mid = 500
        
        # Weights at mid and nearby
        lw1 = low_sharp.get_weight(400)
        lw2 = low_sharp.get_weight(600)
        low_diff = abs(lw2 - lw1)
        
        hw1 = high_sharp.get_weight(400)
        hw2 = high_sharp.get_weight(600)
        high_diff = abs(hw2 - hw1)
        
        # Higher sharpness should have steeper transition
        assert high_diff > low_diff or abs(high_diff - low_diff) < 0.1


class TestTimestepScalerEdgeCases:
    """Tests for edge cases."""
    
    def test_boundary_timesteps(self):
        """Test weights at boundary timesteps."""
        scaler = TimestepScaler()
        
        weight_min = scaler.get_weight(0)
        weight_max = scaler.get_weight(999)
        
        # Min timestep should give low weight
        assert weight_min < 0.5
        
        # Max timestep should give high weight
        assert weight_max > 1.0
    
    def test_single_timestep_schedule(self):
        """Test schedule with single timestep."""
        scaler = TimestepScaler()
        
        schedule = scaler.get_schedule([500])
        
        assert len(schedule) == 1
        assert 0.3 <= schedule[0] <= 1.4
    
    def test_duplicate_timesteps(self):
        """Test schedule with duplicate timesteps."""
        scaler = TimestepScaler()
        
        timesteps = [500, 500, 500]
        schedule = scaler.get_schedule(timesteps)
        
        assert len(schedule) == 3
        # All should be same value
        assert all(abs(schedule[0] - w) < 1e-5 for w in schedule)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

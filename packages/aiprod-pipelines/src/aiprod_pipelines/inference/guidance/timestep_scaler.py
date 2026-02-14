"""
Timestep-aware guidance scaling.

Maps denoising timesteps to guidance strength multipliers.
Higher noise (early steps) get stronger guidance; lower noise (late steps) get weaker.

Classes:
  - TimestepScaler: Smooth scaling curve from timestep to multiplier
"""

from typing import List

import torch


class TimestepScaler:
    """
    Maps denoising timesteps to guidance multipliers.
    
    Key insight: Guidance strength should adapt to noise level.
    - Early steps (high noise, t→1000): Need strong guidance (1.2-1.4x)
    - Mid steps (medium noise, t→500): Baseline guidance (0.9-1.1x)
    - Late steps (low noise, t→0): Weak guidance (0.3-0.5x)
    
    Uses sigmoid-based S-curve for smooth transitions.
    """
    
    def __init__(
        self,
        high_noise_multiplier: float = 1.3,
        mid_noise_multiplier: float = 1.0,
        low_noise_multiplier: float = 0.35,
        total_timesteps: int = 1000,
        sharpness: float = 12.0,
    ):
        """
        Initialize timestep scaler.
        
        Args:
            high_noise_multiplier: Multiplier at high noise (early steps)
            mid_noise_multiplier: Multiplier at medium noise (middle steps)
            low_noise_multiplier: Multiplier at low noise (late steps)
            total_timesteps: Total number of denoising timesteps (default: 1000)
            sharpness: Steepness of sigmoid curve (higher = sharper transition)
        """
        self.high_noise_multiplier = high_noise_multiplier
        self.mid_noise_multiplier = mid_noise_multiplier
        self.low_noise_multiplier = low_noise_multiplier
        self.total_timesteps = total_timesteps
        self.sharpness = sharpness
    
    def get_weight(self, timestep: int) -> float:
        """
        Get guidance multiplier for a specific timestep.
        
        Args:
            timestep: Current denoising timestep [0, total_timesteps]
        
        Returns:
            Guidance multiplier [low_noise_multiplier, high_noise_multiplier]
        
        Example:
            >>> scaler = TimestepScaler()
            >>> scaler.get_weight(0)     # Low noise → ~0.35
            >>> scaler.get_weight(500)   # Mid noise → ~1.0
            >>> scaler.get_weight(999)   # High noise → ~1.3
        """
        # Normalize timestep to [0, 1] where 1 = high noise, 0 = low noise
        t_norm = timestep / self.total_timesteps
        
        # S-curve using sigmoid
        # sigmoid(sharpness * (x - 0.5)) produces smooth transition at x=0.5
        import math
        sigmoid_val = 1.0 / (1.0 + math.exp(-self.sharpness * (t_norm - 0.5)))
        
        # Map sigmoid output [0, 1] to multiplier range
        # High noise (t_norm→1, sigmoid→1): → high_noise_multiplier
        # Low noise (t_norm→0, sigmoid→0): → low_noise_multiplier
        multiplier = (
            self.low_noise_multiplier +
            (sigmoid_val * (self.high_noise_multiplier - self.low_noise_multiplier))
        )
        
        return multiplier
    
    def get_schedule(self, timesteps: List[int]) -> List[float]:
        """
        Get guidance multipliers for a list of timesteps.
        
        Args:
            timesteps: List of timesteps (typically from scheduler)
        
        Returns:
            List of guidance multipliers
        
        Example:
            >>> scaler = TimestepScaler()
            >>> timesteps = [999, 950, 900, ..., 50, 0]
            >>> schedule = scaler.get_schedule(timesteps)
            >>> len(schedule) == len(timesteps)
        """
        return [self.get_weight(t) for t in timesteps]
    
    def get_schedule_tensor(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get guidance multipliers as tensor (for batched operations).
        
        Args:
            timesteps: Tensor of timesteps [batch] or [batch, 1]
        
        Returns:
            Tensor of multipliers matching timesteps shape
        """
        # Normalize
        t_norm = timesteps.float() / self.total_timesteps
        
        # Sigmoid
        sigmoid_val = torch.sigmoid(self.sharpness * (t_norm - 0.5))
        
        # Scale to multiplier range
        multiplier = (
            self.low_noise_multiplier +
            (sigmoid_val * (self.high_noise_multiplier - self.low_noise_multiplier))
        )
        
        return multiplier
    
    def visualize(self, num_points: int = 100) -> tuple:
        """
        Generate visualization data for the scaling curve.
        
        Args:
            num_points: Number of points to sample
        
        Returns:
            Tuple of (timesteps, multipliers) for plotting
        """
        timesteps = [int(i * self.total_timesteps / num_points) for i in range(num_points)]
        multipliers = self.get_schedule(timesteps)
        return timesteps, multipliers
    
    def __repr__(self) -> str:
        return (
            f"TimestepScaler("
            f"high={self.high_noise_multiplier:.2f}, "
            f"mid={self.mid_noise_multiplier:.2f}, "
            f"low={self.low_noise_multiplier:.2f})"
        )


class AdaptiveTimestepScaler(TimestepScaler):
    """
    Extended timestep scaler that adapts based on prompt complexity.
    
    Simple prompts: More aggressive scaling (favor late-step detail)
    Complex prompts: Conservative scaling (maintain guidance throughout)
    """
    
    def __init__(self, **kwargs):
        """Initialize with same parameters as TimestepScaler."""
        super().__init__(**kwargs)
        self.complexity_based_adjustment = True
    
    def get_weight_adaptive(self, timestep: int, complexity: float) -> float:
        """
        Get timestep weight adjusted for prompt complexity.
        
        Args:
            timestep: Current denoising timestep
            complexity: Prompt complexity [0-1]
        
        Returns:
            Adjusted guidance multiplier
        
        Logic:
            Simple prompts (complexity→0): More aggressive scaling
            Complex prompts (complexity→1): Less aggressive scaling
        """
        base_weight = self.get_weight(timestep)
        
        # Adjustment: pull complex prompts toward 1.0 (less variation)
        complexity_factor = 0.5 + (0.5 * complexity)  # [0.5, 1.0]
        
        adjusted = (base_weight - 1.0) * complexity_factor + 1.0
        
        return adjusted

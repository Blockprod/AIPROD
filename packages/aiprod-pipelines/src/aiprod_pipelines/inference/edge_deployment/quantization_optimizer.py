"""
Quantization Optimizer for Edge

Advanced quantization techniques optimized for edge deployment.
Handles post-training quantization, quantization-aware training, and mixed-precision.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class QuantizationType(Enum):
    """Types of quantization."""
    SYMMETRIC = "symmetric"  # Symmetric around zero
    ASYMMETRIC = "asymmetric"  # Asymmetric with offset
    LOGARITHMIC = "logarithmic"  # Logarithmic scaling
    LEARNED = "learned"  # Learnable quantization parameters


class QuantizationLevel(Enum):
    """Quantization precision levels."""
    INT4 = 4  # 4-bit integers
    INT8 = 8  # 8-bit integers
    INT16 = 16  # 16-bit integers
    FP8 = -8  # 8-bit floating point


@dataclass
class QuantizationStats:
    """Statistics for quantization."""
    min_value: float
    max_value: float
    mean_value: float
    std_dev: float
    outlier_threshold: float = 3.0  # 3-sigma
    

@dataclass
class QuantizationScheme:
    """Quantization scheme for a layer."""
    layer_name: str
    quantization_type: QuantizationType
    quantization_level: QuantizationLevel
    scale: float = 1.0
    zero_point: int = 0
    per_channel: bool = False  # Per-channel vs per-layer
    preserve_symmetry: bool = False


class PostTrainingQuantizer:
    """Post-training quantization without retraining."""
    
    def __init__(self, calibration_samples: int = 1000):
        self.calibration_samples = calibration_samples
        self.layer_stats: Dict[str, QuantizationStats] = {}
    
    def calibrate(self, model_layers: List[str], data_loader) -> None:
        """Calibrate quantization parameters."""
        # Collect activation statistics
        for layer_name in model_layers:
            # Simulate gathering statistics
            stats = QuantizationStats(
                min_value=-1.0,
                max_value=1.0,
                mean_value=0.0,
                std_dev=0.5,
            )
            self.layer_stats[layer_name] = stats
    
    def quantize_layer(
        self,
        layer_name: str,
        weights: List[float],
        quantization_level: QuantizationLevel,
    ) -> Tuple[List[int], float, int]:
        """Quantize layer weights."""
        if layer_name not in self.layer_stats:
            return weights, 1.0, 0
        
        stats = self.layer_stats[layer_name]
        
        # Compute quantization parameters
        range_val = stats.max_value - stats.min_value
        if quantization_level == QuantizationLevel.INT8:
            quant_range = 255
        elif quantization_level == QuantizationLevel.INT4:
            quant_range = 15
        else:
            quant_range = 65535
        
        scale = range_val / quant_range
        zero_point = int(-stats.min_value / scale)
        
        # Quantize
        quantized = []
        for w in weights:
            q = int((w - stats.min_value) / scale + 0.5)
            q = max(0, min(quant_range, q))
            quantized.append(q)
        
        return quantized, scale, zero_point


class MixedPrecisionQuantizer:
    """Mixed-precision quantization - different precision for different layers."""
    
    def __init__(self):
        self.layer_precision: Dict[str, QuantizationLevel] = {}
    
    def select_precision_per_layer(
        self,
        model_structure: Dict[str, Dict],
        target_model_size_mb: float,
        current_model_size_mb: float,
    ) -> Dict[str, QuantizationLevel]:
        """Select appropriate precision for each layer."""
        
        # Assign higher precision to important layers, lower to others
        compression_ratio = target_model_size_mb / current_model_size_mb
        
        for layer_name, layer_info in model_structure.items():
            importance = layer_info.get("importance", 0.5)
            
            if importance > 0.7:
                # Important layer: use INT8
                self.layer_precision[layer_name] = QuantizationLevel.INT8
            elif importance > 0.3:
                # Medium important: use INT4
                self.layer_precision[layer_name] = QuantizationLevel.INT4
            else:
                # Less important: aggressive quantization
                self.layer_precision[layer_name] = QuantizationLevel.INT4
        
        return self.layer_precision
    
    def estimate_model_size(self, layer_sizes: Dict[str, float]) -> float:
        """Estimate model size after mixed-precision quantization."""
        total_size = 0.0
        
        for layer_name, original_size in layer_sizes.items():
            precision = self.layer_precision.get(layer_name, QuantizationLevel.INT8)
            
            if precision == QuantizationLevel.INT8:
                compressed_size = original_size * (8 / 32)
            elif precision == QuantizationLevel.INT4:
                compressed_size = original_size * (4 / 32)
            else:
                compressed_size = original_size
            
            total_size += compressed_size
        
        return total_size


class QuantizationAwareTrainer:
    """Simulates quantization during training for better accuracy."""
    
    def __init__(self, quantization_scheme: QuantizationScheme):
        self.scheme = quantization_scheme
        self.training_loss_history: List[float] = []
    
    def simulate_quantization_training(
        self,
        training_iterations: int = 100,
    ) -> Dict[str, float]:
        """Simulate QAT training."""
        # Training metrics
        metrics = {
            "initial_loss": 2.5,
            "final_loss": 1.8,
            "accuracy_retention": 0.98,
            "convergence_iterations": 50,
        }
        
        return metrics


class DynamicQuantizer:
    """Dynamic quantization that adapts to input ranges."""
    
    def __init__(self):
        self.activation_ranges: Dict[str, Tuple[float, float]] = {}
    
    def observe_activation_range(self, layer_name: str, min_val: float, max_val: float) -> None:
        """Observe activation range during inference."""
        if layer_name not in self.activation_ranges:
            self.activation_ranges[layer_name] = (min_val, max_val)
        else:
            current_min, current_max = self.activation_ranges[layer_name]
            self.activation_ranges[layer_name] = (
                min(current_min, min_val),
                max(current_max, max_val),
            )
    
    def get_adaptive_quantization_scheme(self, layer_name: str) -> QuantizationScheme:
        """Get quantization scheme adapted to observed activations."""
        if layer_name in self.activation_ranges:
            min_val, max_val = self.activation_ranges[layer_name]
            range_val = max_val - min_val
            
            # Adapt precision based on range
            if range_val < 0.1:
                # Small range: can use lower precision
                precision = QuantizationLevel.INT4
            elif range_val < 1.0:
                precision = QuantizationLevel.INT8
            else:
                precision = QuantizationLevel.INT16
        else:
            precision = QuantizationLevel.INT8
        
        return QuantizationScheme(
            layer_name=layer_name,
            quantization_type=QuantizationType.ASYMMETRIC,
            quantization_level=precision,
            per_channel=True,
        )

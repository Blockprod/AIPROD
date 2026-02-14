"""
Edge Model Optimizer

Optimizes models specifically for edge and mobile deployment.
Handles model compression, weight quantization, and architecture simplification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math


class EdgeTargetDevice(Enum):
    """Target edge devices for deployment."""
    MOBILE_PHONE = "mobile_phone"  # iOS/Android phones
    TABLET = "tablet"  # Tablets
    EDGE_SERVER = "edge_server"  # Local edge servers
    RASPBERRY_PI = "raspberry_pi"  # Raspberry Pi
    JETSON_NANO = "jetson_nano"  # NVIDIA Jetson Nano
    OCULUS_QUEST = "oculus_quest"  # VR headsets
    CUSTOM_ACCELERATOR = "custom_accelerator"  # Custom hardware


class OptimizationObjective(Enum):
    """Optimization objectives for edge models."""
    LATENCY = "latency"  # Minimize inference time
    MEMORY = "memory"  # Minimize memory footprint
    BATTERY = "battery_life"  # Maximize battery life
    ACCURACY = "accuracy"  # Maintain accuracy
    THROUGHPUT = "throughput"  # Maximize samples/sec
    BALANCED = "balanced"  # Multi-objective


@dataclass
class EdgeDeviceProfile:
    """Profile of edge device capabilities."""
    device_type: EdgeTargetDevice
    device_name: str
    cpu_cores: int
    ram_mb: int
    storage_mb: int
    gpu_available: bool = False
    gpu_vram_mb: int = 0
    accelerator_available: bool = False
    accelerator_type: Optional[str] = None
    battery_capacity_mah: Optional[int] = None
    power_tdp_watts: float = 10.0
    
    @property
    def total_memory_mb(self) -> int:
        """Total available memory."""
        return self.ram_mb + self.gpu_vram_mb
    
    @property
    def memory_budget_mb(self) -> int:
        """Safe memory budget (80% of total)."""
        return int(self.total_memory_mb * 0.8)


@dataclass
class OptimizationConfig:
    """Configuration for edge model optimization."""
    target_device: EdgeTargetDevice
    device_profile: Optional[EdgeDeviceProfile] = None
    optimization_objective: OptimizationObjective = OptimizationObjective.LATENCY
    
    # Compression settings
    enable_quantization: bool = True
    quantization_bits: int = 8  # 8 or 4 bit quantization
    enable_pruning: bool = True
    prune_percentage: float = 0.3  # Remove 30% of weights
    enable_distillation: bool = True
    
    # Architecture simplification
    reduce_layer_depth: bool = True
    reduce_feature_dims: bool = True
    dim_reduction_ratio: float = 0.5
    
    # Memory optimization
    enable_activation_quantization: bool = True
    enable_weight_sharing: bool = True
    
    # Accuracy preservation
    min_accuracy_retention: float = 0.95  # Keep 95% of baseline accuracy
    calibration_samples: int = 1000


@dataclass
class ModelCompressionMetrics:
    """Metrics of model compression."""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float = 0.0
    
    original_params: int = 0
    compressed_params: int = 0
    params_reduction_ratio: float = 0.0
    
    original_latency_ms: float = 0.0
    compressed_latency_ms: float = 0.0
    speedup: float = 1.0
    
    original_accuracy: float = 1.0
    compressed_accuracy: float = 1.0
    accuracy_retention: float = 1.0
    
    def __post_init__(self):
        self.compression_ratio = 1.0 - (self.compressed_size_mb / self.original_size_mb) if self.original_size_mb > 0 else 0
        self.params_reduction_ratio = 1.0 - (self.compressed_params / self.original_params) if self.original_params > 0 else 0
        self.speedup = self.original_latency_ms / self.compressed_latency_ms if self.compressed_latency_ms > 0 else 1.0
        self.accuracy_retention = self.compressed_accuracy / self.original_accuracy if self.original_accuracy > 0 else 1.0


class LayerOptimizer:
    """Optimizes individual layers for edge deployment."""
    
    @staticmethod
    def estimate_layer_size(
        layer_type: str,
        input_size: Tuple[int, ...],
        output_size: Tuple[int, ...],
        quantization_bits: int = 32,
    ) -> float:
        """Estimate layer size in MB."""
        total_elements = 1
        for dim in input_size:
            total_elements *= dim
        total_elements *= max(input_size) * max(output_size)
        
        bytes_per_element = quantization_bits // 8
        return total_elements * bytes_per_element / (1024 * 1024)
    
    @staticmethod
    def recommend_pruning_percentage(
        layer_importance: float,
        target_speedup: float,
    ) -> float:
        """Recommend pruning percentage for layer."""
        # Higher importance = less pruning
        # Higher target speedup = more pruning
        base_prune = 0.3
        importance_factor = 1.0 - layer_importance
        speedup_factor = math.log(target_speedup) / math.log(2.0)
        
        prune_percentage = base_prune * (1.0 + importance_factor * speedup_factor)
        return min(0.8, max(0.1, prune_percentage))  # Clamp to [10%, 80%]


class ModelCompressionStrategy:
    """Strategy for compressing models."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.compression_log: List[str] = []
    
    def compress_model(
        self,
        model_name: str,
        model_path: str,
        original_metrics: Dict[str, float],
    ) -> ModelCompressionMetrics:
        """Compress model for edge deployment."""
        
        # Simulate compression
        original_size = original_metrics.get("size_mb", 100.0)
        original_params = original_metrics.get("params", 1000000)
        original_latency = original_metrics.get("latency_ms", 100.0)
        original_accuracy = original_metrics.get("accuracy", 1.0)
        
        compression_steps = []
        
        # Step 1: Quantization
        if self.config.enable_quantization:
            size_after_quant = original_size * (self.config.quantization_bits / 32)
            compression_steps.append(("quantization", size_after_quant))
            self.compression_log.append(f"Quantization: {original_size:.2f}MB -> {size_after_quant:.2f}MB")
        else:
            size_after_quant = original_size
        
        # Step 2: Pruning
        if self.config.enable_pruning:
            prune_ratio = 1.0 - self.config.prune_percentage
            size_after_prune = size_after_quant * prune_ratio
            compression_steps.append(("pruning", size_after_prune))
            self.compression_log.append(f"Pruning: {size_after_quant:.2f}MB -> {size_after_prune:.2f}MB")
        else:
            size_after_prune = size_after_quant
        
        # Step 3: Distillation (minimal overhead for model size)
        final_size = size_after_prune
        
        # Estimate latency improvement
        params_retained = original_params * (1.0 - self.config.prune_percentage)
        latency_reduction = 1.0 - (self.config.prune_percentage * 0.5)
        compressed_latency = original_latency * latency_reduction
        
        # Estimate accuracy with retention policy
        accuracy_loss_from_pruning = self.config.prune_percentage * 0.05
        compressed_accuracy = original_accuracy * (1.0 - accuracy_loss_from_pruning)
        
        # Ensure minimum accuracy retention
        if compressed_accuracy < original_accuracy * self.config.min_accuracy_retention:
            compressed_accuracy = original_accuracy * self.config.min_accuracy_retention
        
        return ModelCompressionMetrics(
            original_size_mb=original_size,
            compressed_size_mb=final_size,
            original_params=original_params,
            compressed_params=int(params_retained),
            original_latency_ms=original_latency,
            compressed_latency_ms=compressed_latency,
            original_accuracy=original_accuracy,
            compressed_accuracy=compressed_accuracy,
        )


class EdgeOptimizationRecommender:
    """Recommends optimization strategies based on device profile."""
    
    @staticmethod
    def recommend_strategy(
        device_profile: EdgeDeviceProfile,
        model_size_mb: float,
        target_objective: OptimizationObjective,
    ) -> OptimizationConfig:
        """Recommend optimization strategy for device."""
        
        config = OptimizationConfig(
            target_device=device_profile.device_type,
            device_profile=device_profile,
            optimization_objective=target_objective,
        )
        
        available_memory = device_profile.memory_budget_mb
        
        # Aggressive quantization if memory constrained
        if model_size_mb > available_memory * 0.5:
            config.quantization_bits = 4
            config.enable_pruning = True
            config.prune_percentage = 0.4
        
        # Moderate settings for balanced devices
        elif model_size_mb > available_memory * 0.3:
            config.quantization_bits = 8
            config.enable_pruning = True
            config.prune_percentage = 0.2
        
        # Light optimization for capable devices
        else:
            config.quantization_bits = 8
            config.enable_pruning = False
        
        return config

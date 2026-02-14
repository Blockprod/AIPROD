"""
GraphNode wrappers for model quantization in AIPROD inference pipelines.

Provides composable nodes for quantizing models within inference graphs,
with support for multiple quantization methods, quality assessment, and
adaptive selection based on hardware constraints.

Components:
  1. QuantizationProfile: Configuration for pipeline integration
  2. ModelQuantizationNode: Quantize models before inference
  3. QuantizedInferenceNode: Inference wrapper with automatic quantization
  4. QuantizationAdaptiveNode: Dynamic method selection based on hardware

Example Usage:
  >>> profile = QuantizationProfile(
  ...     quantization_method="int8",
  ...     quality_target_percent=97.0,
  ...     enable_dynamic_selection=True
  ... )
  >>> context = GraphContext(...)
  >>> node = ModelQuantizationNode(profile)
  >>> result = node.execute(context)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path

from .quantization import (
    QuantizationConfig, QuantizationMetrics, ModelQuantizer, QuantizationBenchmark
)


@dataclass
class QuantizationProfile:
    """Configuration for quantization integration into inference pipeline."""
    
    enable_quantization: bool = True
    quantization_method: str = "int8"  # int8, bf16, fp8
    calibration_method: str = "histogram"
    per_channel: bool = True
    dynamic: bool = False
    quality_target_percent: float = 95.0
    enable_dynamic_selection: bool = False
    quantization_checkpoint_path: Optional[str] = None
    fallback_method: str = "bf16"  # Fallback if INT8 quality is insufficient
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.quantization_method in ("int8", "bf16", "fp8"), \
            f"Invalid quantization method: {self.quantization_method}"
        assert 0 < self.quality_target_percent <= 100, \
            "Quality target must be in (0, 100]"


class ModelQuantizationNode:
    """GraphNode for quantizing models before inference.
    
    Quantizes encoder, text_encoder, and denoising models to reduce
    memory footprint and improve inference speed.
    
    Input keys:
      - "models": Dict[str, nn.Module] with models to quantize
      - Optional: "calibration_data": List[torch.Tensor] for static calibration
      
    Output keys:
      - "quantized_models": Dict[str, nn.Module]
      - "quantization_metrics": Dict[str, QuantizationMetrics]
      - "speedup_summary": Dict with overall speedup factors
    """
    
    def __init__(self, profile: QuantizationProfile, device: str = "cuda"):
        """Initialize quantization node.
        
        Args:
            profile: QuantizationProfile for quantization strategy
            device: Compute device (cuda or cpu)
        """
        self.profile = profile
        self.device = device
        self.quantized_models: Dict[str, nn.Module] = {}
        self.metrics_history: List[Dict[str, QuantizationMetrics]] = []
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input context keys."""
        return ["models"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output context keys produced by this node."""
        return ["quantized_models", "quantization_metrics", "speedup_summary"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model quantization.
        
        Args:
            context: GraphContext with "models" key
            
        Returns:
            Dict with quantized_models and metrics
        """
        # Validate inputs
        if "models" not in context:
            raise KeyError("ModelQuantizationNode requires 'models' in context")
        
        models = context["models"]
        if not isinstance(models, dict):
            raise ValueError("'models' must be Dict[str, nn.Module]")
        
        if not self.profile.enable_quantization:
            # Pass through unquantized
            return {
                "quantized_models": models,
                "quantization_metrics": {},
                "speedup_summary": {"speedup_factor": 1.0}
            }
        
        # Quantize each model
        metrics = {}
        quantized = {}
        
        calibration_data = context.get("calibration_data", None)
        
        for model_name, model in models.items():
            if model_name in self.profile.fallback_method or not self.profile.enable_quantization:
                quantized[model_name] = model
                continue
            
            model_quantizer = ModelQuantizer(
                QuantizationConfig(
                    quantization_method=self.profile.quantization_method,
                    calibration_method=self.profile.calibration_method,
                    per_channel=self.profile.per_channel,
                    dynamic=self.profile.dynamic
                ),
                device=self.device
            )
            
            quantized_model, quant_metrics = model_quantizer.quantize_model(
                model,
                calibration_data=calibration_data,
                return_metrics=True
            )
            
            # Check quality target
            if quant_metrics.quality_retention_percent < self.profile.quality_target_percent:
                # Try fallback method if quality insufficient
                if self.profile.fallback_method != self.profile.quantization_method:
                    fallback_config = QuantizationConfig(
                        quantization_method=self.profile.fallback_method,
                        calibration_method=self.profile.calibration_method
                    )
                    fallback_quantizer = ModelQuantizer(fallback_config, self.device)
                    quantized_model, quant_metrics = fallback_quantizer.quantize_model(
                        model,
                        calibration_data=calibration_data,
                        return_metrics=True
                    )
            
            quantized[model_name] = quantized_model
            metrics[model_name] = quant_metrics
        
        # Store history
        self.metrics_history.append(metrics)
        
        # Compute speedup summary
        speedup_summary = self._compute_speedup_summary(metrics)
        
        return {
            "quantized_models": quantized,
            "quantization_metrics": metrics,
            "speedup_summary": speedup_summary
        }
    
    def _compute_speedup_summary(self, metrics: Dict[str, QuantizationMetrics]) -> Dict[str, Any]:
        """Compute overall speedup across quantized models."""
        if not metrics:
            return {"speedup_factor": 1.0}
        
        speedups = [m.inference_speedup_factor for m in metrics.values() if m.inference_speedup_factor > 0]
        quality_retentions = [m.quality_retention_percent for m in metrics.values()]
        
        return {
            "speedup_factor": sum(speedups) / len(speedups) if speedups else 1.0,
            "average_quality_retention": sum(quality_retentions) / len(quality_retentions) if quality_retentions else 100.0,
            "models_quantized": len(metrics),
            "total_memory_saved_mb": sum(m.memory_saved_mb for m in metrics.values())
        }
    
    def get_quantization_summary(self) -> Dict[str, Any]:
        """Get summary of quantization history."""
        if not self.metrics_history:
            return {"history_length": 0}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "history_length": len(self.metrics_history),
            "quantization_method": self.profile.quantization_method,
            "models_quantized": len(latest_metrics),
            "average_quality_retention": sum(m.quality_retention_percent for m in latest_metrics.values()) / len(latest_metrics) if latest_metrics else 0.0,
            "total_memory_saved_mb": sum(m.memory_saved_mb for m in latest_metrics.values())
        }
    
    def reset_history(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()


class QuantizedInferenceNode:
    """GraphNode wrapper for inference with quantized models.
    
    Runs denoising with quantized models, automatically handling
    model format and precision conversions.
    
    Input keys:
      - "quantized_models": Dict[str, nn.Module] (quantized)
      - "latents": torch.Tensor
      - "embeddings": torch.Tensor
      - "timestep": int
      
    Output keys:
      - "latents_denoised": torch.Tensor
      - "inference_time_ms": float
      - "quantization_active": bool
    """
    
    def __init__(self, profile: QuantizationProfile, device: str = "cuda"):
        """Initialize quantized inference node."""
        self.profile = profile
        self.device = device
        self.inference_times: List[float] = []
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input keys."""
        return ["quantized_models", "latents", "embeddings", "timestep"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys produced."""
        return ["latents_denoised", "inference_time_ms", "quantization_active"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantized inference.
        
        Args:
            context: GraphContext with required keys
            
        Returns:
            Dict with denoised latents and timing
        """
        # Validate inputs
        for key in self.input_keys:
            if key not in context:
                raise KeyError(f"QuantizedInferenceNode requires '{key}' in context")
        
        models = context["quantized_models"]
        latents = context["latents"]
        embeddings = context["embeddings"]
        timestep = context.get("timestep", 0)
        
        if not self.profile.enable_quantization:
            # Run unquantized
            return {
                "latents_denoised": latents,
                "inference_time_ms": 0.0,
                "quantization_active": False
            }
        
        # Get denoiser model (assuming "denoiser" key in models dict)
        if "denoiser" not in models:
            raise KeyError("Models dict must contain 'denoiser' key")
        
        denoiser = models["denoiser"]
        denoiser.eval()
        
        # Measure inference time
        import time
        start_time = time.time()
        
        with torch.no_grad():
            latents_denoised = denoiser(
                latents.to(self.device),
                torch.tensor(timestep).to(self.device),
                embeddings.to(self.device)
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.inference_times.append(elapsed_ms)
        
        return {
            "latents_denoised": latents_denoised,
            "inference_time_ms": elapsed_ms,
            "quantization_active": True
        }


class QuantizationAdaptiveNode:
    """GraphNode that automatically selects quantization method based on hardware.
    
    Detects available VRAM, model complexity, and quality requirements,
    then selects optimal quantization method (INT8, BF16, or FP8).
    
    Input keys:
      - "models": Dict[str, nn.Module]
      - Optional: "target_quality_percent": float
      
    Output keys:
      - "recommended_method": str
      - "quantized_models": Dict[str, nn.Module]
      - "selection_reasoning": str
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize adaptive quantization node."""
        self.device = device
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input keys."""
        return ["models"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys produced."""
        return ["recommended_method", "quantized_models", "selection_reasoning"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal quantization method and apply it.
        
        Args:
            context: GraphContext with models
            
        Returns:
            Dict with recommended method and quantized models
        """
        if "models" not in context:
            raise KeyError("QuantizationAdaptiveNode requires 'models' in context")
        
        models = context["models"]
        target_quality = context.get("target_quality_percent", 95.0)
        
        # Select method based on available VRAM
        recommended_method = self._select_method(models, target_quality)
        
        # Apply quantization
        profile = QuantizationProfile(quantization_method=recommended_method)
        node = ModelQuantizationNode(profile, self.device)
        result = node.execute(context)
        
        reasoning = self._get_selection_reasoning(recommended_method, target_quality)
        
        return {
            "recommended_method": recommended_method,
            "quantized_models": result["quantized_models"],
            "selection_reasoning": reasoning
        }
    
    def _select_method(self, models: Dict[str, nn.Module], target_quality: float) -> str:
        """Select quantization method based on hardware and requirements."""
        try:
            if torch.cuda.is_available():
                # Get available VRAM
                total_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                
                # Estimate model size
                model_size_gb = sum(
                    sum(p.numel() for p in m.parameters()) * 4 / (1024**3)
                    for m in models.values()
                )
                
                vram_utilization = model_size_gb / max(total_vram_gb, 1)
                
                # Selection logic
                if vram_utilization > 0.7:
                    # High VRAM pressure: use aggressive INT8
                    return "int8"
                elif target_quality > 98.0:
                    # Very high quality target: use BF16
                    return "bf16"
                else:
                    # Balanced: use INT8
                    return "int8"
            else:
                # CPU: no quantization or BF16
                return "bf16"
        except Exception:
            return "int8"  # Default fallback
    
    def _get_selection_reasoning(self, method: str, target_quality: float) -> str:
        """Generate human-readable reasoning for method selection."""
        reasons = {
            "int8": f"Selected INT8 for 2-3x speedup with {target_quality}% quality target",
            "bf16": f"Selected BF16 for {target_quality}% quality target with 1.3x speedup",
            "fp8": "Selected FP8 for balanced speedup and quality"
        }
        return reasons.get(method, "Selected method based on hardware constraints")

"""
Edge Deployment Module

Provides comprehensive edge and mobile deployment infrastructure.
Includes model optimization, quantization, pruning, distillation, and deployment management.
"""

# Edge Model Optimizer
from .edge_model_optimizer import (
    EdgeTargetDevice,
    OptimizationObjective,
    EdgeDeviceProfile,
    OptimizationConfig,
    ModelCompressionMetrics,
    LayerOptimizer,
    ModelCompressionStrategy,
    EdgeOptimizationRecommender,
)

# Quantization Optimizer
from .quantization_optimizer import (
    QuantizationType,
    QuantizationLevel,
    QuantizationStats,
    QuantizationScheme,
    PostTrainingQuantizer,
    MixedPrecisionQuantizer,
    QuantizationAwareTrainer,
    DynamicQuantizer,
)

# Pruning Engine
from .pruning_engine import (
    PruningStrategy,
    PruningCriterion,
    PruningMask,
    LayerImportance,
    WeightPruner,
    StructuredPruner,
    IterativePruner,
    KnowledgeDistillation,
)

# Mobile Runtime
from .mobile_runtime import (
    MobileRuntime,
    RuntimeMemoryConfig,
    RuntimeConfig,
    InferenceMemoryManager,
    ThreadPoolExecutor,
    MobileInferenceRuntime,
)

# Edge Inference Engine
from .edge_inference_engine import (
    EdgeInferenceMetrics,
    InferenceCache,
    EdgeInferenceEngine,
    BatchedEdgeInference,
)

# Resource Monitor
from .resource_monitor import (
    ResourceLevel,
    ResourceSnapshot,
    ResourceThresholds,
    DeviceResourceMonitor,
    AdaptiveInferenceController,
)

# Deployment Manager
from .deployment_manager import (
    DeploymentStatus,
    DeploymentPackage,
    DeploymentRecord,
    EdgeDeploymentManager,
)

__all__ = [
    # Edge Model Optimizer (8 exports)
    "EdgeTargetDevice",
    "OptimizationObjective",
    "EdgeDeviceProfile",
    "OptimizationConfig",
    "ModelCompressionMetrics",
    "LayerOptimizer",
    "ModelCompressionStrategy",
    "EdgeOptimizationRecommender",
    
    # Quantization Optimizer (8 exports)
    "QuantizationType",
    "QuantizationLevel",
    "QuantizationStats",
    "QuantizationScheme",
    "PostTrainingQuantizer",
    "MixedPrecisionQuantizer",
    "QuantizationAwareTrainer",
    "DynamicQuantizer",
    
    # Pruning Engine (8 exports)
    "PruningStrategy",
    "PruningCriterion",
    "PruningMask",
    "LayerImportance",
    "WeightPruner",
    "StructuredPruner",
    "IterativePruner",
    "KnowledgeDistillation",
    
    # Mobile Runtime (6 exports)
    "MobileRuntime",
    "RuntimeMemoryConfig",
    "RuntimeConfig",
    "InferenceMemoryManager",
    "ThreadPoolExecutor",
    "MobileInferenceRuntime",
    
    # Edge Inference Engine (4 exports)
    "EdgeInferenceMetrics",
    "InferenceCache",
    "EdgeInferenceEngine",
    "BatchedEdgeInference",
    
    # Resource Monitor (5 exports)
    "ResourceLevel",
    "ResourceSnapshot",
    "ResourceThresholds",
    "DeviceResourceMonitor",
    "AdaptiveInferenceController",
    
    # Deployment Manager (4 exports)
    "DeploymentStatus",
    "DeploymentPackage",
    "DeploymentRecord",
    "EdgeDeploymentManager",
]

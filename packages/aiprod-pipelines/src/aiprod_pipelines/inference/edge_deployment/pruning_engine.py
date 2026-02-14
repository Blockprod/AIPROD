"""
Model Pruning Engine

Implements structured and unstructured pruning for edge deployment.
Removes redundant weights while maintaining accuracy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class PruningStrategy(Enum):
    """Pruning strategies."""
    UNSTRUCTURED = "unstructured"  # Individual weight pruning
    STRUCTURED = "structured"  # Channel/filter pruning
    LAYER_WISE = "layer_wise"  # Entire layers
    ITERATIVE = "iterative"  # Gradual pruning


class PruningCriterion(Enum):
    """Weight importance criteria for pruning."""
    MAGNITUDE = "magnitude"  # Magnitude-based
    GRADIENT = "gradient"  # Gradient-based
    FISHER = "fisher"  # Fisher information
    TAYLOR = "taylor"  # First-order Taylor expansion
    LEARNABLE = "learnable"  # Learned masks


@dataclass
class PruningMask:
    """Binary mask for pruned weights."""
    layer_name: str
    mask: List[bool]  # True = keep, False = prune
    pruned_count: int = 0
    pruning_ratio: float = 0.0


@dataclass
class LayerImportance:
    """Importance score for layer."""
    layer_name: str
    importance_score: float  # 0-1, higher = more important
    criterion_used: PruningCriterion
    

class WeightPruner:
    """Prunes weights using various criteria."""
    
    def __init__(self, strategy: PruningStrategy):
        self.strategy = strategy
        self.layer_importance: Dict[str, LayerImportance] = {}
    
    def compute_layer_importance(
        self,
        layer_name: str,
        weights: List[float],
        gradients: Optional[List[float]] = None,
        criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
    ) -> LayerImportance:
        """Compute importance score for layer."""
        
        if criterion == PruningCriterion.MAGNITUDE:
            # Average absolute weight magnitude
            avg_magnitude = sum(abs(w) for w in weights) / len(weights)
            importance = min(1.0, avg_magnitude)
        elif criterion == PruningCriterion.GRADIENT and gradients:
            # Average absolute gradient
            avg_grad = sum(abs(g) for g in gradients) / len(gradients)
            importance = min(1.0, avg_grad)
        else:
            importance = 0.5
        
        importance_obj = LayerImportance(
            layer_name=layer_name,
            importance_score=importance,
            criterion_used=criterion,
        )
        
        self.layer_importance[layer_name] = importance_obj
        return importance_obj
    
    def generate_pruning_mask(
        self,
        weights: List[float],
        pruning_ratio: float,
        criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
    ) -> PruningMask:
        """Generate pruning mask for weights."""
        
        if criterion == PruningCriterion.MAGNITUDE:
            # Sort by magnitude
            indexed_weights = [(i, abs(w)) for i, w in enumerate(weights)]
            indexed_weights.sort(key=lambda x: x[1])
            
            # Prune smallest weights
            prune_count = int(len(weights) * pruning_ratio)
            prune_indices = set(idx for idx, _ in indexed_weights[:prune_count])
            
            mask = [i not in prune_indices for i in range(len(weights))]
        else:
            # Default: keep all
            mask = [True] * len(weights)
            prune_indices = set()
        
        return PruningMask(
            layer_name="layer",
            mask=mask,
            pruned_count=len(prune_indices),
            pruning_ratio=len(prune_indices) / len(weights),
        )


class StructuredPruner:
    """Prunes entire channels or filters."""
    
    def __init__(self):
        self.channel_importance: Dict[str, List[float]] = {}
    
    def compute_channel_importance(
        self,
        layer_name: str,
        filters: List[List[float]],  # List of channel/filter weights
    ) -> List[float]:
        """Compute importance for each channel."""
        channel_importance = []
        
        for channel_weights in filters:
            # Average magnitude of channel
            importance = sum(abs(w) for w in channel_weights) / len(channel_weights) if channel_weights else 0
            channel_importance.append(importance)
        
        self.channel_importance[layer_name] = channel_importance
        return channel_importance
    
    def select_important_channels(
        self,
        layer_name: str,
        num_channels: int,
        reduction_ratio: float = 0.3,
    ) -> List[int]:
        """Select important channels to keep."""
        if layer_name not in self.channel_importance:
            return list(range(num_channels))
        
        importance_scores = self.channel_importance[layer_name]
        
        # Sort by importance
        indexed_importance = [(i, score) for i, score in enumerate(importance_scores)]
        indexed_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top channels
        keep_count = int(num_channels * (1 - reduction_ratio))
        important_channels = [idx for idx, _ in indexed_importance[:keep_count]]
        
        return sorted(important_channels)


class IterativePruner:
    """Iteratively prunes model with retraining."""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.pruning_history: List[Dict] = []
    
    def iterative_prune(
        self,
        initial_model_size_mb: float,
        target_model_size_mb: float,
        eval_function: callable,
    ) -> Dict:
        """Iteratively prune and retrain."""
        
        current_size = initial_model_size_mb
        current_accuracy = 1.0
        pruning_iteration = 0
        
        while current_size > target_model_size_mb and pruning_iteration < self.max_iterations:
            # Calculate pruning ratio for this iteration
            size_ratio = target_model_size_mb / initial_model_size_mb
            prune_ratio = 1.0 - (size_ratio ** (1.0 / (self.max_iterations - pruning_iteration)))
            prune_ratio = min(0.3, prune_ratio)  # Cap at 30% per iteration
            
            # Simulate pruning
            current_size = current_size * (1 - prune_ratio)
            
            # Simulate accuracy drop
            accuracy_drop = prune_ratio * 0.05
            current_accuracy = max(0.95, current_accuracy - accuracy_drop)
            
            iteration_info = {
                "iteration": pruning_iteration,
                "model_size_mb": current_size,
                "accuracy": current_accuracy,
                "pruning_ratio": prune_ratio,
            }
            self.pruning_history.append(iteration_info)
            
            pruning_iteration += 1
        
        return {
            "final_size_mb": current_size,
            "final_accuracy": current_accuracy,
            "iterations": pruning_iteration,
            "history": self.pruning_history,
        }


class KnowledgeDistillation:
    """Knowledge distillation from teacher to student model."""
    
    def __init__(self, temperature: float = 4.0):
        self.temperature = temperature
        self.distillation_loss_history: List[float] = []
    
    def compute_distillation_loss(
        self,
        teacher_logits: List[float],
        student_logits: List[float],
        true_labels: List[int],
        alpha: float = 0.5,  # Weight for distillation loss
    ) -> float:
        """Compute knowledge distillation loss."""
        # Simplified distillation loss
        # In practice would use softmax and cross-entropy
        
        # KL divergence between teacher and student
        kl_loss = sum(abs(t - s) for t, s in zip(teacher_logits, student_logits)) / len(teacher_logits)
        
        # Standard cross-entropy on true labels
        ce_loss = 1.0 - sum(1 for t, s in zip(true_labels, student_logits) if t == s) / len(true_labels)
        
        # Weighted combination
        total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
        self.distillation_loss_history.append(total_loss)
        
        return total_loss
    
    def estimate_student_performance(
        self,
        teacher_accuracy: float,
        distillation_iterations: int = 100,
    ) -> float:
        """Estimate student model accuracy after distillation."""
        # Models typically retain 90-98% of teacher accuracy
        retention_ratio = 0.95 - (0.02 * math.exp(-distillation_iterations / 50))
        return teacher_accuracy * retention_ratio


import math

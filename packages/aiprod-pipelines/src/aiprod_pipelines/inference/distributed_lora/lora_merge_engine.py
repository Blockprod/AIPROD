"""
LoRA Merge Engine and Model Composition

Enables merging of multiple LoRA adapters, hierarchical model inheritance,
and dynamic adapter composition for inference.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time


class MergeStrategy(Enum):
    """Strategies for merging LoRA adapters"""
    LINEAR = "linear"  # Simple linear combination
    HIERARCHICAL = "hierarchical"  # Parent-child merge
    MIXTURE = "mixture"  # Mixture of experts / attention-based
    INTERPOLATION = "interpolation"  # Smooth interpolation
    STACKING = "stacking"  # Stack adapters sequentially


@dataclass
class AdapterWeight:
    """Weight for combining adapters"""
    adapter_id: str
    weight: float = 1.0
    scaling_factor: float = 1.0
    
    def normalized_weight(self, total_weight: float) -> float:
        """Get normalized weight"""
        return (self.weight * self.scaling_factor) / max(total_weight, 1e-8)


@dataclass
class CompositionPlan:
    """Plan for composing multiple adapters"""
    plan_id: str
    adapters: List[AdapterWeight]
    strategy: MergeStrategy
    description: str = ""
    created_at: float = field(default_factory=time.time)
    
    @property
    def total_weight(self) -> float:
        """Total weight across adapters"""
        return sum(a.weight * a.scaling_factor for a in self.adapters)


class LoRAMergeEngine:
    """Merges multiple LoRA adapters into single effective adapter"""
    
    def __init__(self):
        self.composition_cache: Dict[str, Dict[str, Any]] = {}
        self.merge_history: List[Dict[str, Any]] = []
    
    def merge_adapters(self, adapters: List[Tuple[str, Any, float]], 
                       strategy: MergeStrategy = MergeStrategy.LINEAR) -> Dict[str, Any]:
        """
        Merge multiple LoRA adapters
        
        Args:
            adapters: List of (adapter_id, weights, weight_coefficient)
            strategy: Merge strategy
        
        Returns:
            Merged adapter weights
        """
        if not adapters:
            return {}
        
        if strategy == MergeStrategy.LINEAR:
            return self._merge_linear(adapters)
        elif strategy == MergeStrategy.HIERARCHICAL:
            return self._merge_hierarchical(adapters)
        elif strategy == MergeStrategy.MIXTURE:
            return self._merge_mixture(adapters)
        else:
            return self._merge_linear(adapters)
    
    def _merge_linear(self, adapters: List[Tuple[str, Any, float]]) -> Dict[str, Any]:
        """Linear combination of adapters"""
        if not adapters:
            return {}
        
        merged = {}
        total_weight = sum(w for _, _, w in adapters)
        
        for adapter_id, weights, coeff in adapters:
            normalized_coeff = coeff / max(total_weight, 1e-8)
            for param_name, param_value in weights.items():
                if param_name not in merged:
                    merged[param_name] = 0
                # Simple addition with normalization
                if hasattr(param_value, '__len__'):
                    try:
                        merged[param_name] += sum(param_value) * normalized_coeff / len(param_value)
                    except:
                        merged[param_name] += normalized_coeff
                else:
                    merged[param_name] += param_value * normalized_coeff
        
        return merged
    
    def _merge_hierarchical(self, adapters: List[Tuple[str, Any, float]]) -> Dict[str, Any]:
        """Hierarchical merge with parent-child relationships"""
        # Sort by ID to establish hierarchy
        sorted_adapters = sorted(adapters, key=lambda x: x[0])
        
        # Use first as base, blend others
        base_id, base_weights, base_coeff = sorted_adapters[0]
        merged = {k: v for k, v in base_weights.items()}
        
        # Blend in other adapters with decreasing weight
        for idx, (adapter_id, weights, coeff) in enumerate(sorted_adapters[1:], 1):
            blend_factor = 1.0 / (idx + 1)
            for param_name, param_value in weights.items():
                if param_name in merged:
                    if hasattr(param_value, '__len__'):
                        merged[param_name] = (merged[param_name] * (1 - blend_factor) + 
                                            sum(param_value) * blend_factor / len(param_value))
        
        return merged
    
    def _merge_mixture(self, adapters: List[Tuple[str, Any, float]]) -> Dict[str, Any]:
        """Mixture of experts merge with gating"""
        # Simple attention-based gating
        weights = [w for _, _, w in adapters]
        total = sum(weights)
        softmax_weights = [w / total for w in weights]
        
        merged = {}
        for (adapter_id, adapter_weights, _), gate_weight in zip(adapters, softmax_weights):
            for param_name, param_value in adapter_weights.items():
                if param_name not in merged:
                    merged[param_name] = 0
                if hasattr(param_value, '__len__'):
                    try:
                        merged[param_name] += sum(param_value) * gate_weight / len(param_value)
                    except:
                        merged[param_name] += gate_weight
                else:
                    merged[param_name] += param_value * gate_weight
        
        return merged
    
    def save_composition(self, plan: CompositionPlan, merged_weights: Dict[str, Any]):
        """Save composition plan and merged weights"""
        self.composition_cache[plan.plan_id] = {
            "plan": plan,
            "merged_weights": merged_weights,
            "created_at": time.time()
        }
        
        self.merge_history.append({
            "plan_id": plan.plan_id,
            "strategy": plan.strategy.value,
            "num_adapters": len(plan.adapters),
            "timestamp": time.time()
        })
    
    def load_composition(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Load cached composition"""
        return self.composition_cache.get(plan_id)
    
    def estimate_merge_size(self, adapters: List[Tuple[str, Any, float]]) -> int:
        """Estimate size of merged adapter"""
        if not adapters:
            return 0
        
        # Base size from first adapter
        _, first_weights, _ = adapters[0]
        base_size = sum(self._count_parameters(v) for v in first_weights.values())
        
        # Merged size is same as individual (no multiplication)
        return base_size


class ModelInheritance:
    """Manages hierarchical LoRA model inheritance"""
    
    def __init__(self):
        self.inheritance_graph: Dict[str, List[str]] = {}  # model_id -> list of child_ids
        self.model_lineage: Dict[str, str] = {}  # model_id -> parent_id
    
    def register_inheritance(self, child_id: str, parent_id: str):
        """Register parent-child inheritance relationship"""
        self.model_lineage[child_id] = parent_id
        
        if parent_id not in self.inheritance_graph:
            self.inheritance_graph[parent_id] = []
        self.inheritance_graph[parent_id].append(child_id)
    
    def get_parent_model(self, model_id: str) -> Optional[str]:
        """Get parent model"""
        return self.model_lineage.get(model_id)
    
    def get_child_models(self, model_id: str) -> List[str]:
        """Get all child models"""
        return self.inheritance_graph.get(model_id, [])
    
    def get_ancestors(self, model_id: str) -> List[str]:
        """Get all ancestor models up to root"""
        ancestors = []
        current = model_id
        
        while current in self.model_lineage:
            parent = self.model_lineage[current]
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def get_descendants(self, model_id: str, depth: int = -1) -> List[str]:
        """Get all descendant models"""
        descendants = []
        to_visit = [(model_id, 0)]
        
        while to_visit:
            current, current_depth = to_visit.pop(0)
            
            if depth >= 0 and current_depth >= depth:
                continue
            
            children = self.get_child_models(current)
            descendants.extend(children)
            
            for child in children:
                to_visit.append((child, current_depth + 1))
        
        return descendants
    
    def compute_distance(self, model_id_1: str, model_id_2: str) -> int:
        """Compute distance between two models in inheritance tree"""
        ancestors_1 = set(self.get_ancestors(model_id_1) + [model_id_1])
        ancestors_2 = set(self.get_ancestors(model_id_2) + [model_id_2])
        
        # Find lowest common ancestor
        common = ancestors_1 & ancestors_2
        if not common:
            return -1  # No relationship
        
        lca = list(common)[0]  # Arbitrary common ancestor
        
        # Distance = distance to LCA + distance from LCA
        dist1 = len(self.get_ancestors(model_id_1)) - len(
            [a for a in self.get_ancestors(model_id_1) if a == lca] or 
            self.get_ancestors(lca)
        )
        dist2 = len(self.get_ancestors(model_id_2)) - len(
            [a for a in self.get_ancestors(model_id_2) if a == lca] or 
            self.get_ancestors(lca)
        )
        
        return dist1 + dist2


class AdapterComposer:
    """Composes adapters dynamically for inference"""
    
    def __init__(self, merge_engine: LoRAMergeEngine):
        self.merge_engine = merge_engine
        self.active_compositions: Dict[str, Dict[str, Any]] = {}
    
    def compose_for_task(self, task_id: str, adapter_ids: List[str],
                         weights: Optional[List[float]] = None,
                         strategy: MergeStrategy = MergeStrategy.LINEAR) -> str:
        """
        Compose adapters for specific task
        
        Returns:
            Composition ID
        """
        if weights is None:
            weights = [1.0] * len(adapter_ids)
        
        adapters = [(aid, {}, w) for aid, w in zip(adapter_ids, weights)]
        plan = CompositionPlan(
            plan_id=f"task_{task_id}_{len(self.active_compositions)}",
            adapters=[AdapterWeight(aid, w) for aid, w in zip(adapter_ids, weights)],
            strategy=strategy,
            description=f"Composed for task {task_id}"
        )
        
        merged = self.merge_engine.merge_adapters(adapters, strategy)
        self.merge_engine.save_composition(plan, merged)
        
        return plan.plan_id
    
    def get_composition(self, composition_id: str) -> Optional[Dict[str, Any]]:
        """Get active composition"""
        return self.merge_engine.load_composition(composition_id)
    
    def _count_parameters(self, value: Any) -> int:
        """Count parameters"""
        if hasattr(value, '__len__'):
            try:
                return len(value)
            except:
                return 1
        return 1

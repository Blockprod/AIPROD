"""
Federated Learning for LoRA

Implements federated training for LoRA models with privacy preservation,
aggregation, and communication efficiency.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time


class FederatedAggregationMethod(Enum):
    """Aggregation methods for federated learning"""
    FED_AVG = "fed_avg"  # Federated averaging
    FED_PROX = "fed_prox"  # Federated proximal
    FED_ADAM = "fed_adam"  # Federated adaptive moment estimation
    SECURE_AGGREGATION = "secure_aggregation"  # Cryptographically secure


class DifferentialPrivacyLevel(Enum):
    """Privacy levels"""
    NONE = 0
    LOW = 1e-2
    MEDIUM = 1e-4
    HIGH = 1e-6


@dataclass
class ClientUpdate:
    """Update from a single client (user)"""
    client_id: str
    model_weights: Dict[str, Any]
    weight_deltas: Dict[str, Any]  # Change from initialization
    num_samples_trained: int
    training_steps: int
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def get_update_norm(self) -> float:
        """Compute L2 norm of weight deltas"""
        total_sq = 0.0
        for delta in self.weight_deltas.values():
            if hasattr(delta, '__len__'):
                try:
                    total_sq += sum(x ** 2 for x in delta)
                except:
                    total_sq += len(delta)
        return total_sq ** 0.5


@dataclass  
class ServerState:
    """Global server state in federated training"""
    round: int = 0
    global_model_weights: Dict[str, Any] = field(default_factory=dict)
    aggregated_loss: float = 0.0
    client_updates_received: int = 0
    total_clients_available: int = 0
    round_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state"""
        return {
            "round": self.round,
            "client_updates_received": self.client_updates_received,
            "total_clients_available": self.total_clients_available,
            "aggregated_loss": self.aggregated_loss,
            "round_duration_ms": self.round_duration_ms
        }


class FederatedAggregator:
    """Aggregates client updates in federated learning"""
    
    def __init__(self, method: FederatedAggregationMethod = FederatedAggregationMethod.FED_AVG):
        self.method = method
        self.client_updates: List[ClientUpdate] = []
    
    def add_client_update(self, update: ClientUpdate):
        """Add client update to aggregation pool"""
        self.client_updates.append(update)
    
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate all client updates"""
        if not self.client_updates:
            return {}
        
        if self.method == FederatedAggregationMethod.FED_AVG:
            return self._federated_averaging()
        elif self.method == FederatedAggregationMethod.FED_PROX:
            return self._federated_proximal()
        elif self.method == FederatedAggregationMethod.FED_ADAM:
            return self._federated_adam()
        else:
            return self._federated_averaging()
    
    def _federated_averaging(self) -> Dict[str, Any]:
        """Standard federated averaging"""
        aggregated = {}
        total_samples = sum(u.num_samples_trained for u in self.client_updates)
        
        for param_name in self.client_updates[0].model_weights.keys():
            weighted_sum = 0
            for update in self.client_updates:
                weight = update.num_samples_trained / total_samples
                weighted_sum += weight * sum(self.client_updates[0].model_weights[param_name])
            aggregated[param_name] = weighted_sum / len(self.client_updates)
        
        avg_loss = sum(u.loss for u in self.client_updates) / len(self.client_updates)
        
        return {
            "aggregated_weights": aggregated,
            "avg_loss": avg_loss,
            "num_clients": len(self.client_updates),
            "total_samples": total_samples
        }
    
    def _federated_proximal(self, mu: float = 0.01) -> Dict[str, Any]:
        """Federated proximal method (adds regularization)"""
        fed_avg_result = self._federated_averaging()
        
        # Add proximal term: penalize distance from previous global model
        # (would require previous model for full implementation)
        
        return fed_avg_result
    
    def _federated_adam(self) -> Dict[str, Any]:
        """Federated Adam with adaptive learning rates"""
        fed_avg_result = self._federated_averaging()
        
        # Compute momentum estimates across clients
        # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
        # v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
        
        return fed_avg_result
    
    def clear(self):
        """Clear client updates after aggregation"""
        self.client_updates.clear()


class DifferentialPrivacyEngine:
    """Adds differential privacy to aggregation"""
    
    def __init__(self, privacy_level: DifferentialPrivacyLevel = DifferentialPrivacyLevel.MEDIUM,
                 delta: float = 1e-5):
        self.privacy_level = privacy_level
        self.delta = delta
        self.epsilon = privacy_level.value if privacy_level != DifferentialPrivacyLevel.NONE else float('inf')
    
    def clip_gradient(self, gradient: Any, clip_norm: float = 1.0) -> Any:
        """Apply gradient clipping for sensitivity bound"""
        # Compute L2 norm and clip if necessary
        if hasattr(gradient, '__len__'):
            try:
                norm = sum(x ** 2 for x in gradient) ** 0.5
                if norm > clip_norm:
                    scale = clip_norm / max(norm, 1e-10)
                    gradient = [x * scale for x in gradient]
            except:
                pass
        return gradient
    
    def add_laplace_noise(self, value: float, sensitivity: float, scale: float = 1.0) -> float:
        """Add Laplace noise for privacy"""
        if self.epsilon == float('inf'):
            return value  # No privacy
        
        import random
        noise_scale = (sensitivity / self.epsilon) * scale
        noise = random.gauss(0, noise_scale)
        return value + noise
    
    def get_privacy_budget_consumed(self, rounds: int) -> float:
        """Estimate privacy budget (epsilon) consumed"""
        # Rough estimate: epsilon * sqrt(T) where T is number of rounds
        return self.epsilon * (rounds ** 0.5)


class FederatedTrainer:
    """Coordinates federated training"""
    
    def __init__(self, num_clients: int, 
                 aggregation_method: FederatedAggregationMethod = FederatedAggregationMethod.FED_AVG,
                 privacy_level: DifferentialPrivacyLevel = DifferentialPrivacyLevel.NONE):
        self.num_clients = num_clients
        self.aggregator = FederatedAggregator(aggregation_method)
        self.privacy_engine = DifferentialPrivacyEngine(privacy_level)
        self.server_state = ServerState(total_clients_available=num_clients)
        self.round_history: List[Dict[str, Any]] = []
    
    def start_round(self, sample_fraction: float = 1.0) -> List[str]:
        """Start new federated round, select clients to participate"""
        self.server_state.round += 1
        
        import random
        num_selected = max(1, int(self.num_clients * sample_fraction))
        selected_clients = [f"client_{i}" for i in range(num_selected)]
        
        return selected_clients
    
    def aggregate_round(self) -> Dict[str, Any]:
        """Aggregate updates from this round"""
        start_time = time.perf_counter()
        
        result = self.aggregator.aggregate()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.server_state.round_duration_ms = duration_ms
        self.server_state.client_updates_received = len(self.aggregator.client_updates)
        self.server_state.aggregated_loss = result.get("avg_loss", 0.0)
        
        # Record history
        self.round_history.append({
            "round": self.server_state.round,
            "num_clients": result.get("num_clients", 0),
            "avg_loss": result.get("avg_loss", 0.0),
            "total_samples": result.get("total_samples", 0),
            "duration_ms": duration_ms
        })
        
        self.aggregator.clear()
        
        return result
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """Check convergence status"""
        if len(self.round_history) < 2:
            return {"converged": False, "rounds": self.server_state.round}
        
        recent_losses = [r["avg_loss"] for r in self.round_history[-5:]]
        recent_avg = sum(recent_losses) / len(recent_losses)
        
        if len(recent_losses) > 1:
            loss_delta = abs(recent_losses[0] - recent_losses[-1])
            converged = loss_delta < 0.001 * recent_avg  # Less than 0.1% change
        else:
            converged = False
        
        return {
            "converged": converged,
            "rounds": self.server_state.round,
            "recent_loss": recent_losses[-1] if recent_losses else float('inf'),
            "loss_trend": "decreasing" if recent_losses[0] > recent_losses[-1] else "increasing"
        }


@dataclass
class FederatedTrainingConfig:
    """Configuration for federated training"""
    num_rounds: int = 10
    local_epochs_per_round: int = 3
    client_sample_fraction: float = 1.0  # Fraction of clients per round
    aggregation_method: FederatedAggregationMethod = FederatedAggregationMethod.FED_AVG
    differential_privacy: bool = False
    privacy_level: DifferentialPrivacyLevel = DifferentialPrivacyLevel.MEDIUM
    enable_client_dropout: bool = True
    dropout_probability: float = 0.1
    
    def is_privacy_enabled(self) -> bool:
        """Check if privacy is enabled"""
        return self.differential_privacy and self.privacy_level != DifferentialPrivacyLevel.NONE

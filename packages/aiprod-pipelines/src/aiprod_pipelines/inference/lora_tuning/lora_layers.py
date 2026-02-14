"""LoRA layer implementations for parameter-efficient fine-tuning.

Provides:
- LoRA layer wrapper
- Low-rank weight decomposition
- Adapter composition
- Scaled addition/subtraction
"""

from typing import Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAWeight:
    """LoRA weight representation."""
    
    name: str                      # Layer name
    in_features: int              # Input dimension
    out_features: int             # Output dimension
    rank: int                      # LoRA rank
    alpha: float = 1.0            # Scaling factor


class LoRALinear(nn.Module):
    """LoRA adapter for linear layers.
    
    Replaces standard linear layer with low-rank decomposition:
    output = x @ W + (x @ A) @ B * (alpha / rank)
    """
    
    def __init__(
        self,
        base_module: nn.Linear,
        rank: int = 8,
        alpha: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            base_module: Original linear layer
            rank: Low-rank dimension
            alpha: Scaling factor
            dropout_rate: Dropout on LoRA input
        """
        super().__init__()
        
        self.base_linear = base_module
        self.in_features = base_module.in_features
        self.out_features = base_module.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create LoRA weights
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )
        
        # Initialize
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Freeze base module
        self.base_linear.requires_grad_(False)
        
        logger.debug(
            f"LoRA Linear: {self.in_features} x {self.out_features}, "
            f"rank={rank}, alpha={alpha}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output = base(x) + LoRA(x)
        """
        # Base layer forward
        base_out = self.base_linear(x)
        
        # LoRA forward: (x @ A) @ B * scaling
        if self.dropout:
            x_lora = self.dropout(x)
        else:
            x_lora = x
        
        lora_out = (x_lora @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    """LoRA adapter for 2D convolutional layers."""
    
    def __init__(
        self,
        base_module: nn.Conv2d,
        rank: int = 8,
        alpha: float = 1.0,
    ):
        """
        Initialize LoRA conv2d layer.
        
        Args:
            base_module: Original conv2d layer
            rank: Low-rank dimension
            alpha: Scaling factor
        """
        super().__init__()
        
        self.base_conv = base_module
        self.in_channels = base_module.in_channels
        self.out_channels = base_module.out_channels
        self.kernel_size = base_module.kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights: decompose conv kernel
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_channels, 1, 1)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_channels, rank, 1, 1)
        )
        
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        self.base_conv.requires_grad_(False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA adaptation."""
        base_out = self.base_conv(x)
        
        # LoRA: apply low-rank conv adaptation
        lora_out = torch.nn.functional.conv2d(
            x,
            self.lora_A,
            padding=0,
        )
        lora_out = torch.nn.functional.conv2d(
            lora_out,
            self.lora_B,
            padding=0,
        )
        
        # Pad to match base output shape if needed
        if lora_out.shape != base_out.shape:
            pad = (
                (base_out.shape[-1] - lora_out.shape[-1]) // 2,
                (base_out.shape[-1] - lora_out.shape[-1]) // 2,
                (base_out.shape[-2] - lora_out.shape[-2]) // 2,
                (base_out.shape[-2] - lora_out.shape[-2]) // 2,
            )
            lora_out = torch.nn.functional.pad(lora_out, pad)
        
        return base_out + lora_out * self.scaling


class LoRAAdapter(nn.Module):
    """Adapter that manages LoRA layers for a model."""
    
    def __init__(self):
        """Initialize LoRA adapter."""
        super().__init__()
        self.lora_layers = nn.ModuleDict()
        self.scale = 1.0
    
    def add_lora_linear(
        self,
        name: str,
        base_module: nn.Linear,
        rank: int = 8,
        alpha: float = 1.0,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Add LoRA to a linear layer.
        
        Args:
            name: Layer name
            base_module: Original linear layer
            rank: LoRA rank
            alpha: Scaling factor
            dropout_rate: Dropout rate
        """
        lora_layer = LoRALinear(base_module, rank, alpha, dropout_rate)
        self.lora_layers[name] = lora_layer
    
    def add_lora_conv2d(
        self,
        name: str,
        base_module: nn.Conv2d,
        rank: int = 8,
        alpha: float = 1.0,
    ) -> None:
        """Add LoRA to conv2d layer."""
        lora_layer = LoRAConv2d(base_module, rank, alpha)
        self.lora_layers[name] = lora_layer
    
    def get_parameters(self) -> torch.nn.ParameterList:
        """Get LoRA parameters."""
        params = []
        for layer in self.lora_layers.values():
            if isinstance(layer, (LoRALinear, LoRAConv2d)):
                params.append(layer.lora_A)
                params.append(layer.lora_B)
        return torch.nn.ParameterList(params)
    
    def get_parameter_count(self) -> int:
        """Count total LoRA parameters."""
        total = 0
        for layer in self.lora_layers.values():
            if hasattr(layer, "lora_A"):
                total += layer.lora_A.numel()
            if hasattr(layer, "lora_B"):
                total += layer.lora_B.numel()
        return total
    
    def forward(self):
        """Adapter doesn't have forward (used as wrapper)."""
        pass


class LoRAComposer:
    """Manages multiple LoRA adapters and their composition."""
    
    def __init__(self, composition_mode: str = "sequential"):
        """
        Initialize LoRA composer.
        
        Args:
            composition_mode: "sequential", "parallel", "gated", "conditional"
        """
        self.composition_mode = composition_mode
        self.adapters = {}
        self.gating_network = None
    
    def add_adapter(self, name: str, adapter: LoRAAdapter) -> None:
        """Add named adapter."""
        self.adapters[name] = adapter
    
    def remove_adapter(self, name: str) -> None:
        """Remove named adapter."""
        if name in self.adapters:
            del self.adapters[name]
    
    def get_active_adapters(self) -> list:
        """Get list of active adapter names."""
        return list(self.adapters.keys())
    
    def compose_outputs(
        self,
        outputs: dict,
        weights: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compose outputs from multiple adapters.
        
        Args:
            outputs: Dict mapping adapter name to output tensor
            weights: Optional weights for each adapter
            
        Returns:
            Composed output
        """
        if self.composition_mode == "sequential":
            # Apply sequentially (last one wins)
            result = None
            for name, output in outputs.items():
                if result is None:
                    result = output
                else:
                    result = result + output
            return result
        
        elif self.composition_mode == "parallel":
            # Sum all outputs
            return sum(outputs.values())
        
        elif self.composition_mode == "gated":
            # Learned gating
            if weights is None:
                weights = {name: 1.0 / len(outputs) for name in outputs}
            
            result = None
            for name, output in outputs.items():
                weight = weights.get(name, 0.0)
                if result is None:
                    result = output * weight
                else:
                    result = result + output * weight
            return result
        
        else:
            raise ValueError(f"Unknown composition mode: {self.composition_mode}")
    
    def get_total_parameters(self) -> int:
        """Count total parameters across all adapters."""
        total = 0
        for adapter in self.adapters.values():
            total += adapter.get_parameter_count()
        return total


class LoRAMerger:
    """Merges LoRA weights back into base model."""
    
    @staticmethod
    def merge_linear(
        base_module: nn.Linear,
        lora_layer: LoRALinear,
        scaling: float = 1.0,
    ) -> None:
        """
        Merge LoRA layer into base linear.
        
        Args:
            base_module: Base linear layer
            lora_layer: LoRA linear layer
            scaling: Scaling factor
        """
        with torch.no_grad():
            # Compute merged weight: W' = W + (B @ A) * scaling
            merged_weight = lora_layer.lora_B @ lora_layer.lora_A
            base_module.weight.add_(merged_weight * scaling)
    
    @staticmethod
    def unmerge_linear(
        base_module: nn.Linear,
        lora_layer: LoRALinear,
        scaling: float = 1.0,
    ) -> None:
        """Unmerge LoRA from base (restore original)."""
        with torch.no_grad():
            merged_weight = lora_layer.lora_B @ lora_layer.lora_A
            base_module.weight.sub_(merged_weight * scaling)

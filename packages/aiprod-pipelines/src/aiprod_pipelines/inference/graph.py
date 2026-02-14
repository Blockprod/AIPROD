"""
Core InferenceGraph infrastructure for unified inference execution.

Implements node-based DAG (Direct Acyclic Graph) composition for flexible
pipeline orchestration. Replaces 5+ monolithic pipeline classes with a
single composable system.

Key Classes:
  - GraphNode: Abstract base for all inference nodes
  - GraphContext: Execution context holding shared data
  - InferenceGraph: DAG executor with topological scheduling
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch


@dataclass
class GraphContext:
    """Execution context holding intermediate and final results."""
    
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dtype: torch.dtype = torch.bfloat16
    
    def __getitem__(self, key: str) -> Any:
        """Get value from context (try outputs first, then inputs)."""
        if key in self.outputs:
            return self.outputs[key]
        if key in self.inputs:
            return self.inputs[key]
        raise KeyError(f"Key '{key}' not found in context")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value in context outputs."""
        self.outputs[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in context."""
        return key in self.outputs or key in self.inputs
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update outputs with new data."""
        self.outputs.update(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def clear_intermediates(self) -> None:
        """Clear intermediate results (keep final outputs)."""
        # Subclasses can override for memory optimization
        pass


class GraphNode(ABC):
    """
    Abstract base class for all inference graph nodes.
    
    Each node represents a single operation in the inference pipeline.
    Nodes receive inputs from context, perform computation, and produce outputs.
    """
    
    def __init__(self, node_id: Optional[str] = None, **kwargs):
        """
        Initialize node.
        
        Args:
            node_id: Unique identifier for this node instance
            **kwargs: Node-specific configuration parameters
        """
        self.node_id = node_id or self.__class__.__name__
        self.config = kwargs
        self._validate_config()
    
    @abstractmethod
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute node logic.
        
        Args:
            context: GraphContext with inputs and shared state
            
        Returns:
            Dict of outputs to add to context
            
        Raises:
            ValueError: If required inputs are missing
            RuntimeError: If execution fails
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate node configuration. Override in subclasses."""
        pass
    
    def _validate_inputs(self, context: GraphContext, required: List[str]) -> None:
        """
        Validate that required inputs exist in context.
        
        Args:
            context: GraphContext to check
            required: List of required input keys
            
        Raises:
            ValueError: If any required input is missing
        """
        missing = [key for key in required if key not in context]
        if missing:
            raise ValueError(
                f"Node '{self.node_id}' missing required inputs: {missing}. "
                f"Available: {set(context.inputs.keys()) | set(context.outputs.keys())}"
            )
    
    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """List of required input keys."""
        pass
    
    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """List of output keys produced by this node."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node_id})"


class InferenceGraph:
    """
    Direct Acyclic Graph (DAG) for composable inference pipelines.
    
    Enables flexible composition of inference steps without code duplication.
    Supports:
    - Custom node arrangement
    - Preset configurations
    - Topological execution ordering
    - Performance optimizations (node fusion, kernel compilation)
    
    Example:
        ```python
        graph = InferenceGraph()
        graph.add_node("encode", TextEncodeNode(...))
        graph.add_node("denoise", DenoiseNode(...))
        graph.add_node("decode", DecodeVideoNode())
        
        graph.connect("encode", "denoise")
        graph.connect("denoise", "decode")
        
        result = graph.run(prompt="...", seed=42)
        ```
    """
    
    def __init__(self, name: str = "inference_graph"):
        """
        Initialize inference graph.
        
        Args:
            name: Graph name for identification
        """
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # node_id → {next_node_ids}
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # node_id → {prev_node_ids}
        self._cached_topo_sort: Optional[List[str]] = None
    
    def add_node(self, node_id: str, node: GraphNode) -> "InferenceGraph":
        """
        Add node to graph.
        
        Args:
            node_id: Unique identifier for node
            node: GraphNode instance
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists in graph")
        
        self.nodes[node_id] = node
        self._cached_topo_sort = None  # Invalidate cache
        return self
    
    def connect(self, from_node: str, to_node: str) -> "InferenceGraph":
        """
        Connect two nodes (create edge from → to).
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If nodes don't exist or edge creates cycle
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found in graph")
        if to_node not in self.nodes:
            raise ValueError(f"Destination node '{to_node}' not found in graph")
        
        # Check for cycles (DFS)
        if self._would_create_cycle(from_node, to_node):
            raise ValueError(
                f"Adding edge {from_node} → {to_node} would create a cycle"
            )
        
        self.edges[from_node].add(to_node)
        self.reverse_edges[to_node].add(from_node)
        self._cached_topo_sort = None  # Invalidate cache
        return self
    
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Execute graph in topological order.
        
        Args:
            **inputs: Input parameters for the graph
            
        Returns:
            Dict of final outputs
            
        Raises:
            ValueError: If execution fails or graph is invalid
            RuntimeError: If node execution fails
        """
        # Validate graph structure
        if not self.nodes:
            raise ValueError("Cannot run empty graph")
        
        # Initialize context
        context = GraphContext(inputs=inputs)
        
        # Execute nodes in topological order
        execution_order = self._topological_sort()
        
        for node_id in execution_order:
            node = self.nodes[node_id]
            
            try:
                outputs = node.execute(context)
                context.update(outputs)
            except Exception as e:
                raise RuntimeError(
                    f"Node '{node_id}' execution failed: {e}"
                ) from e
        
        return context.outputs
    
    def _topological_sort(self) -> List[str]:
        """
        Return nodes in execution order using Kahn's algorithm.
        
        Returns:
            List of node IDs in topological order
            
        Raises:
            ValueError: If graph contains cycles
        """
        if self._cached_topo_sort is not None:
            return self._cached_topo_sort
        
        # Calculate in-degrees
        in_degree = {node_id: len(self.reverse_edges[node_id]) for node_id in self.nodes}
        
        # Initialize queue with zero in-degree nodes
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        
        if not queue:
            raise ValueError(
                "Graph has cycles - no node with in-degree 0. "
                "Ensure graph is a valid DAG."
            )
        
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            # Reduce in-degree for successors
            for next_node in self.edges[node_id]:
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    queue.append(next_node)
        
        if len(result) != len(self.nodes):
            raise ValueError(
                "Graph contains cycles - not all nodes were visited during topological sort"
            )
        
        self._cached_topo_sort = result
        return result
    
    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """
        Check if adding edge would create a cycle using DFS.
        
        Args:
            from_node: Source node
            to_node: Destination node
            
        Returns:
            True if cycle would be created, False otherwise
        """
        # DFS from to_node - if we reach from_node, there's a cycle
        visited = set()
        stack = [to_node]
        
        while stack:
            current = stack.pop()
            if current == from_node:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            stack.extend(self.edges[current])
        
        return False
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate graph structure.
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not self.nodes:
            return False, "Graph is empty"
        
        try:
            self._topological_sort()
        except ValueError as e:
            return False, str(e)
        
        return True, "Graph is valid"
    
    def summary(self) -> str:
        """
        Get human-readable graph summary.
        
        Returns:
            String describing graph structure
        """
        try:
            topo_order = self._topological_sort()
        except ValueError:
            topo_order = list(self.nodes.keys())
        
        lines = [f"InferenceGraph: {self.name}", f"Nodes: {len(self.nodes)}"]
        
        for node_id in topo_order:
            node = self.nodes[node_id]
            inputs = ", ".join(node.input_keys)
            outputs = ", ".join(node.output_keys)
            successors = list(self.edges[node_id])
            
            lines.append(f"  {node_id}: {node.__class__.__name__}")
            lines.append(f"    ├─ Inputs:  [{inputs}]")
            lines.append(f"    ├─ Outputs: [{outputs}]")
            if successors:
                lines.append(f"    └─ Connects to: {successors}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"InferenceGraph(name={self.name}, nodes={len(self.nodes)})"

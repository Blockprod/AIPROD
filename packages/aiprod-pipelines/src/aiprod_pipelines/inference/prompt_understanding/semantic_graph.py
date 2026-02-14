"""Semantic graph construction and traversal.

Builds knowledge graphs from prompt concepts and relationships.
Enables:
- Graph-based reasoning
- Relationship traversal
- Relevance scoring
- Scene composition understanding
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in semantic graph."""
    
    concept: str              # Concept text
    entity_type: str         # Entity type
    importance: float = 0.5  # 0-1 importance
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Node({self.concept}, imp={self.importance:.2f})"


@dataclass
class GraphEdge:
    """Edge in semantic graph."""
    
    source: str              # Source node
    target: str              # Target node
    relation_type: str       # Type of relationship
    strength: float = 0.8    # 0-1 edge weight


class SemanticGraph:
    """Builds and analyzes semantic graphs from concepts.

    Represents:
    - Entities as nodes
    - Relationships as edges
    - Attributes on nodes/edges
    - Multi-hop relationships
    """
    
    def __init__(self):
        """Initialize semantic graph."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[str]] = {}
        self._relevance_cache = {}
    
    def add_node(
        self,
        concept: str,
        entity_type: str,
        importance: float = 0.5,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add node to graph.
        
        Args:
            concept: Concept text
            entity_type: Type of entity
            importance: Importance score 0-1
            attributes: Optional node attributes
        """
        if concept not in self.nodes:
            node = GraphNode(
                concept=concept,
                entity_type=entity_type,
                importance=min(1.0, importance),
                attributes=attributes or {},
            )
            self.nodes[concept] = node
            self.adjacency[concept] = []
            logger.debug(f"Added node: {concept}")
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        strength: float = 0.8,
    ) -> None:
        """
        Add edge to graph.
        
        Args:
            source: Source node
            target: Target node
            relation_type: Type of relationship
            strength: Edge weight 0-1
        """
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"Cannot add edge: nodes {source} or {target} not found")
            return
        
        # Check for duplicate edge
        duplicate = next(
            (e for e in self.edges 
             if e.source == source and e.target == target),
            None
        )
        
        if not duplicate:
            edge = GraphEdge(
                source=source,
                target=target,
                relation_type=relation_type,
                strength=min(1.0, strength),
            )
            self.edges.append(edge)
            
            # Update adjacency
            if target not in self.adjacency[source]:
                self.adjacency[source].append(target)
            
            logger.debug(f"Added edge: {source} -> {target} ({relation_type})")
    
    def get_neighbors(self, concept: str, max_depth: int = 1) -> Dict[str, float]:
        """
        Get neighboring concepts with relevance scores.
        
        Args:
            concept: Source concept
            max_depth: Maximum hops to traverse
            
        Returns:
            Dictionary mapping concept to relevance score
        """
        neighbors = {}
        visited = {concept}
        frontier = [(concept, 1.0, 0)]  # (node, score, depth)
        
        while frontier:
            current, score, depth = frontier.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get adjacent nodes
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # Find edge strength
                    edge_strength = next(
                        (e.strength for e in self.edges
                         if e.source == current and e.target == neighbor),
                        0.8
                    )
                    
                    # Compute neighbor score
                    neighbor_score = score * edge_strength * 0.9 ** depth
                    neighbors[neighbor] = neighbor_score
                    
                    # Add to frontier
                    frontier.append((neighbor, neighbor_score, depth + 1))
        
        # Sort by relevance
        return dict(sorted(neighbors.items(), key=lambda x: x[1], reverse=True))
    
    def compute_graph_density(self) -> float:
        """
        Compute graph density (0-1).
        
        Returns:
            Density score
        """
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1) / 2
        actual_edges = len(self.edges)
        
        return actual_edges / max_edges
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> List[List[str]]:
        """
        Find all paths between two nodes.
        
        Args:
            source: Starting node
            target: Ending node
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is list of nodes)
        """
        if source not in self.nodes or target not in self.nodes:
            return []
        
        paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if len(path) > max_length:
                return
            
            if current == target:
                paths.append(path)
                return
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, target, path + [neighbor], visited)
                    visited.remove(neighbor)
        
        visited = {source}
        dfs(source, target, [source], visited)
        
        return paths
    
    def compute_relevance(
        self,
        concept: str,
        reference_concepts: Optional[List[str]] = None,
    ) -> float:
        """
        Compute relevance of concept.
        
        Args:
            concept: Concept to score
            reference_concepts: Optional reference set (e.g., main subjects)
            
        Returns:
            Relevance score 0-1
        """
        if concept not in self.nodes:
            return 0.0
        
        # Check cache
        cache_key = (concept, tuple(reference_concepts or []))
        if cache_key in self._relevance_cache:
            return self._relevance_cache[cache_key]
        
        node = self.nodes[concept]
        relevance = node.importance
        
        # Boost by proximity to reference concepts
        if reference_concepts:
            for ref_concept in reference_concepts:
                paths = self.find_paths(ref_concept, concept, max_length=3)
                if paths:
                    shortest_path = min(paths, key=len)
                    # Closer = higher relevance boost
                    proximity_boost = 1.0 / (len(shortest_path) - 1)
                    relevance = min(1.0, relevance + proximity_boost * 0.3)
        
        # Cache result
        self._relevance_cache[cache_key] = relevance
        
        return relevance
    
    def get_central_concepts(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get most central (important) concepts.
        
        Args:
            top_k: Number of concepts to return
            
        Returns:
            List of (concept, centrality_score) tuples
        """
        centrality = {}
        
        for concept, node in self.nodes.items():
            # Compute centrality as combination of:
            # 1. Node importance
            # 2. Degree (number of connections)
            # 3. Edge strength
            
            degree = len(self.adjacency.get(concept, []))
            edge_strength = sum(
                e.strength for e in self.edges
                if e.source == concept or e.target == concept
            ) / (max(degree, 1) * 2)
            
            centrality_score = (
                0.5 * node.importance +
                0.3 * min(1.0, degree / 10) +  # Normalize degree
                0.2 * edge_strength
            )
            
            centrality[concept] = centrality_score
        
        # Sort and return top K
        sorted_concepts = sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_concepts[:top_k]
    
    def get_topic_clusters(self, num_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Group concepts into topic clusters (simplified).
        
        Args:
            num_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster ID to concept list
        """
        clusters = {i: [] for i in range(num_clusters)}
        
        # Group by entity type (simplified clustering)
        entity_groups = {}
        for concept, node in self.nodes.items():
            entity_type = node.entity_type
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(concept)
        
        # Distribute groups across clusters
        cluster_idx = 0
        for entity_type, concepts in entity_groups.items():
            for concept in concepts:
                clusters[cluster_idx % num_clusters].append(concept)
            cluster_idx += 1
        
        return clusters
    
    def export_structure(self) -> Dict[str, Any]:
        """
        Export graph structure for visualization/analysis.
        
        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "nodes": [
                {
                    "concept": concept,
                    "entity_type": node.entity_type,
                    "importance": node.importance,
                    "attributes": node.attributes,
                }
                for concept, node in self.nodes.items()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation_type": edge.relation_type,
                    "strength": edge.strength,
                }
                for edge in self.edges
            ],
            "metrics": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "density": self.compute_graph_density(),
                "central_concepts": self.get_central_concepts(5),
            },
        }
    
    def clear(self) -> None:
        """Clear the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self._relevance_cache.clear()
        logger.info("Graph cleared")

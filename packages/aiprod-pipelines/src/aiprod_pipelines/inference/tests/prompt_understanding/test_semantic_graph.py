"""Test suite for semantic graph construction and traversal.

Tests:
- Graph node/edge creation
- Path finding
- Centrality computation
- Relevance scoring
- Graph analysis
"""

import pytest
from aiprod_pipelines.inference.prompt_understanding.semantic_graph import (
    GraphNode,
    GraphEdge,
    SemanticGraph,
)


class TestGraphNode:
    """Test GraphNode dataclass."""
    
    def test_node_creation(self):
        """Test creating graph node."""
        node = GraphNode(
            concept="cat",
            entity_type="OBJECT",
            importance=0.9,
        )
        
        assert node.concept == "cat"
        assert node.entity_type == "OBJECT"
        assert node.importance == 0.9
    
    def test_node_with_attributes(self):
        """Test node with attributes."""
        attrs = {"color": "orange", "size": "small"}
        node = GraphNode(
            concept="cat",
            entity_type="OBJECT",
            attributes=attrs,
        )
        
        assert node.attributes["color"] == "orange"


class TestGraphEdge:
    """Test GraphEdge dataclass."""
    
    def test_edge_creation(self):
        """Test creating graph edge."""
        edge = GraphEdge(
            source="cat",
            target="walking",
            relation_type="performs",
            strength=0.9,
        )
        
        assert edge.source == "cat"
        assert edge.target == "walking"
        assert edge.relation_type == "performs"


class TestSemanticGraph:
    """Test SemanticGraph functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create semantic graph."""
        return SemanticGraph()
    
    def test_graph_initialization(self, graph):
        """Test graph setup."""
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_node(self, graph):
        """Test adding node to graph."""
        graph.add_node("cat", "OBJECT", importance=0.9)
        
        assert "cat" in graph.nodes
        assert graph.nodes["cat"].importance == 0.9
    
    def test_add_edge(self, graph):
        """Test adding edge to graph."""
        graph.add_node("cat", "OBJECT")
        graph.add_node("walking", "ACTION")
        graph.add_edge("cat", "walking", "performs", strength=0.9)
        
        assert len(graph.edges) == 1
        assert "walking" in graph.adjacency["cat"]
    
    def test_duplicate_node_prevention(self, graph):
        """Test that duplicate nodes don't get added."""
        graph.add_node("cat", "OBJECT", importance=0.9)
        graph.add_node("cat", "OBJECT", importance=0.8)
        
        # Should still have only one node
        assert len(graph.nodes) == 1
        assert graph.nodes["cat"].importance == 0.9  # First value preserved
    
    def test_invalid_edge(self, graph):
        """Test adding edge with non-existent nodes."""
        graph.add_edge("cat", "nonexistent", "performs")
        
        # Should not add edge
        assert len(graph.edges) == 0
    
    def test_neighbors(self, graph):
        """Test getting neighbors."""
        # Build simple graph: cat -> walking -> quickly
        graph.add_node("cat", "OBJECT", importance=0.9)
        graph.add_node("walking", "ACTION", importance=0.85)
        graph.add_node("quickly", "ADVERB", importance=0.6)
        
        graph.add_edge("cat", "walking", "performs", strength=0.9)
        graph.add_edge("walking", "quickly", "modifies", strength=0.8)
        
        neighbors = graph.get_neighbors("cat", max_depth=2)
        
        assert "walking" in neighbors
        assert "quickly" in neighbors
    
    def test_graph_density(self, graph):
        """Test density computation."""
        # Fully connected triangle
        graph.add_node("a", "X")
        graph.add_node("b", "X")
        graph.add_node("c", "X")
        
        graph.add_edge("a", "b", "rel")
        graph.add_edge("b", "c", "rel")
        graph.add_edge("a", "c", "rel")
        
        density = graph.compute_graph_density()
        
        # Fully connected triangle has density 1.0
        assert density == 1.0
    
    def test_path_finding(self, graph):
        """Test finding paths between nodes."""
        # Build graph: a -> b -> c -> d
        for node in ["a", "b", "c", "d"]:
            graph.add_node(node, "X")
        
        graph.add_edge("a", "b", "rel")
        graph.add_edge("b", "c", "rel")
        graph.add_edge("c", "d", "rel")
        
        paths = graph.find_paths("a", "d")
        
        assert len(paths) > 0
        assert ["a", "b", "c", "d"] in paths
    
    def test_path_finding_multiple(self, graph):
        """Test finding multiple paths."""
        # Build graph with multiple paths
        for node in ["a", "b", "c", "d"]:
            graph.add_node(node, "X")
        
        # Path 1: a -> b -> d
        # Path 2: a -> c -> d
        graph.add_edge("a", "b", "rel")
        graph.add_edge("b", "d", "rel")
        graph.add_edge("a", "c", "rel")
        graph.add_edge("c", "d", "rel")
        
        paths = graph.find_paths("a", "d")
        
        assert len(paths) >= 2
    
    def test_relevance_computation(self, graph):
        """Test relevance scoring."""
        graph.add_node("cat", "OBJECT", importance=0.9)
        graph.add_node("walking", "ACTION", importance=0.85)
        graph.add_node("forest", "LOCATION", importance=0.7)
        
        graph.add_edge("cat", "walking", "performs")
        graph.add_edge("walking", "forest", "in")
        
        # Forest relevance to cat
        relevance = graph.compute_relevance("forest", reference_concepts=["cat"])
        
        # Should boost relevance when connected to reference
        assert relevance > 0.5
    
    def test_central_concepts(self, graph):
        """Test finding central concepts."""
        # Hub-and-spoke: central connected to many
        graph.add_node("central", "X", importance=0.5)
        for i in range(5):
            graph.add_node(f"node_{i}", "X", importance=0.3)
            graph.add_edge("central", f"node_{i}", "rel")
        
        central = graph.get_central_concepts(top_k=3)
        
        # Central should be first
        assert central[0][0] == "central"
    
    def test_topic_clustering(self, graph):
        """Test topic clustering."""
        graph.add_node("cat", "OBJECT")
        graph.add_node("dog", "OBJECT")
        graph.add_node("walking", "ACTION")
        graph.add_node("running", "ACTION")
        
        clusters = graph.get_topic_clusters(num_clusters=2)
        
        assert len(clusters) == 2
        # Objects and actions should be separated
        assert len(clusters[0]) > 0 or len(clusters[1]) > 0
    
    def test_export_structure(self, graph):
        """Test exporting graph structure."""
        graph.add_node("cat", "OBJECT", importance=0.9)
        graph.add_node("walking", "ACTION", importance=0.85)
        graph.add_edge("cat", "walking", "performs", strength=0.9)
        
        structure = graph.export_structure()
        
        assert "nodes" in structure
        assert "edges" in structure
        assert "metrics" in structure
        assert len(structure["nodes"]) == 2
        assert len(structure["edges"]) == 1


class TestGraphIntegration:
    """Integration tests for semantic graph."""
    
    def test_full_graph_construction(self):
        """Test building complete semantic graph."""
        graph = SemanticGraph()
        
        # Build scene graph
        entities = [
            ("cat", "OBJECT", 0.95),
            ("forest", "LOCATION", 0.85),
            ("walking", "ACTION", 0.90),
            ("bright", "QUALITY", 0.60),
            ("sunlight", "ATTRIBUTE", 0.70),
        ]
        
        for concept, entity_type, importance in entities:
            graph.add_node(concept, entity_type, importance=importance)
        
        # Add relationships
        relationships = [
            ("cat", "walking", "performs"),
            ("walking", "forest", "in"),
            ("bright", "sunlight", "describes"),
            ("sunlight", "forest", "illuminates"),
        ]
        
        for source, target, relation in relationships:
            graph.add_edge(source, target, relation)
        
        # Verify structure
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        
        # Test queries
        neighbors = graph.get_neighbors("cat", max_depth=2)
        assert len(neighbors) >= 2
    
    def test_scene_graph_analysis(self):
        """Test analyzing scene from graph."""
        graph = SemanticGraph()
        
        # Build scene: bright cat in forest
        scenes = [
            ("cat", "OBJECT", 0.95),
            ("bright", "QUALITY", 0.7),
            ("forest", "LOCATION", 0.85),
        ]
        
        for concept, entity_type, imp in scenes:
            graph.add_node(concept, entity_type, importance=imp)
        
        graph.add_edge("bright", "cat", "describes")
        graph.add_edge("cat", "forest", "in")
        
        # Analyze
        central = graph.get_central_concepts(top_k=3)
        
        assert len(central) > 0
    
    def test_graph_clearing(self):
        """Test clearing graph."""
        graph = SemanticGraph()
        graph.add_node("cat", "OBJECT")
        
        assert len(graph.nodes) == 1
        
        graph.clear()
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

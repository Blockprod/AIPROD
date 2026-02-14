"""Prompt understanding module for semantic-aware video generation.

Phase 2 Foundation:
- PromptAnalyzer: Parse and analyze prompts
- SemanticTokenizer: Advanced tokenization with roles
- ConceptExtractor: Extract entities and concepts
- SemanticGraph: Build knowledge graphs from concepts

Phase 5 Enhancements:
- SemanticPromptAnalyzer: Advanced semantic analysis
- EntityRecognizer: Fine-grained entity recognition
- PromptEnhancementEngine: Auto-enhancement capabilities
"""

# Phase 2 Foundations
from .prompt_analyzer import (
    PromptToken,
    PromptAnalysisResult,
    PromptAnalyzer,
)

from .semantic_tokenizer import (
    RelationType,
    SemanticRelationship,
    TokenRole,
    EnhancedToken,
    SemanticTokenizer,
)

from .concept_extractor import (
    EntityType as ConceptEntityType,
    Concept,
    ConceptRelationship,
    ConceptExtractor,
)

from .semantic_graph import (
    GraphNode,
    GraphEdge,
    SemanticGraph,
)

# Phase 5 Enhancements
from .semantic_prompt_analyzer import (
    SemanticIntent,
    SemanticToken,
    ConceptCluster,
    SemanticAnalysisResult,
    SemanticPromptAnalyzer,
    SemanticSimilarityMatcher,
)

from .entity_recognition import (
    EntityType,
    Entity,
    EntityCluster,
    EntityRecognizer,
    ObjectRelationshipExtractor,
    SceneUnderstanding,
)

from .prompt_enhancement_engine import (
    EnhancementStrategy,
    PromptEnhancement,
    EnhancedPromptResult,
    PromptEnhancementEngine,
    IterativePromptOptimizer,
)

__all__ = [
    # Phase 2: PromptAnalyzer exports
    "PromptToken",
    "PromptAnalysisResult",
    "PromptAnalyzer",
    
    # Phase 2: SemanticTokenizer exports
    "RelationType",
    "SemanticRelationship",
    "TokenRole",
    "EnhancedToken",
    "SemanticTokenizer",
    
    # Phase 2: ConceptExtractor exports
    "ConceptEntityType",
    "Concept",
    "ConceptRelationship",
    "ConceptExtractor",
    
    # Phase 2: SemanticGraph exports
    "GraphNode",
    "GraphEdge",
    "SemanticGraph",
    
    # Phase 5: SemanticPromptAnalyzer exports (6)
    "SemanticIntent",
    "SemanticToken",
    "ConceptCluster",
    "SemanticAnalysisResult",
    "SemanticPromptAnalyzer",
    "SemanticSimilarityMatcher",
    
    # Phase 5: EntityRecognition exports (6)
    "EntityType",
    "Entity",
    "EntityCluster",
    "EntityRecognizer",
    "ObjectRelationshipExtractor",
    "SceneUnderstanding",
    
    # Phase 5: PromptEnhancement exports (5)
    "EnhancementStrategy",
    "PromptEnhancement",
    "EnhancedPromptResult",
    "PromptEnhancementEngine",
    "IterativePromptOptimizer",
]

"""Test suite for semantic tokenization.

Tests:
- Token role assignment
- Relationship identification
- Dependency structures
- Phrase extraction
- Semantic role labeling
"""

import pytest
from aiprod_pipelines.inference.prompt_understanding.semantic_tokenizer import (
    RelationType,
    SemanticRelationship,
    TokenRole,
    EnhancedToken,
    SemanticTokenizer,
)


class TestTokenRole:
    """Test TokenRole enum."""
    
    def test_role_values(self):
        """Test token role values."""
        assert TokenRole.PROTAGONIST.value == "protagonist"
        assert TokenRole.ACTION.value == "action"
        assert TokenRole.SETTING.value == "setting"


class TestEnhancedToken:
    """Test EnhancedToken dataclass."""
    
    def test_token_creation(self):
        """Test creating enhanced token."""
        token = EnhancedToken(
            text="cat",
            idx=0,
            token_type="NOUN",
            role=TokenRole.PROTAGONIST,
            salience=0.95,
        )
        
        assert token.text == "cat"
        assert token.role == TokenRole.PROTAGONIST
        assert token.salience == 0.95
    
    def test_token_relationships(self):
        """Test token relationship tracking."""
        token = EnhancedToken(
            text="cat",
            idx=0,
            token_type="NOUN",
            role=TokenRole.PROTAGONIST,
        )
        
        token.related_to.append(1)
        token.related_to.append(2)
        
        assert len(token.related_to) == 2


class TestSemanticRelationship:
    """Test SemanticRelationship dataclass."""
    
    def test_relationship_creation(self):
        """Test creating relationship."""
        rel = SemanticRelationship(
            source_idx=0,
            target_idx=1,
            rel_type=RelationType.SUBJECT_VERB,
            strength=0.9,
        )
        
        assert rel.source_idx == 0
        assert rel.target_idx == 1
        assert rel.rel_type == RelationType.SUBJECT_VERB


class TestSemanticTokenizer:
    """Test SemanticTokenizer functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer."""
        return SemanticTokenizer()
    
    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer setup."""
        assert tokenizer is not None
        assert len(tokenizer.role_patterns) > 0
    
    def test_role_assignment(self, tokenizer):
        """Test semantic role assignment."""
        # Protagonist role
        role = tokenizer._assign_role("cat")
        assert role == TokenRole.PROTAGONIST
        
        # Action role
        role = tokenizer._assign_role("walking")
        assert role == TokenRole.ACTION
        
        # Setting role
        role = tokenizer._assign_role("forest")
        assert role == TokenRole.SETTING
    
    def test_pos_tagging(self, tokenizer):
        """Test part-of-speech tagging."""
        # Verb
        pos = tokenizer._get_pos_tag("walking")
        assert pos == "VERB"
        
        # Adverb
        pos = tokenizer._get_pos_tag("slowly")
        assert pos == "ADV"
        
        # Noun (capitalized)
        pos = tokenizer._get_pos_tag("Forest")
        assert pos == "NOUN"
    
    def test_salience_computation(self, tokenizer):
        """Test token salience computation."""
        # Protagonist high salience
        sal = tokenizer._compute_salience("cat", TokenRole.PROTAGONIST)
        assert sal > 0.9
        
        # Connector low salience
        sal = tokenizer._compute_salience("in", TokenRole.CONNECTOR)
        assert sal < 0.3
    
    def test_semantic_tokenization(self, tokenizer):
        """Test complete semantic tokenization."""
        tokens = ["cat", "walking", "forest"]
        
        enhanced_tokens, relationships = tokenizer.tokenize_semantic("test", tokens)
        
        assert len(enhanced_tokens) == 3
        assert all(isinstance(t, EnhancedToken) for t in enhanced_tokens)
    
    def test_relationship_identification(self, tokenizer):
        """Test relationship identification."""
        # Create tokens in semantic relationship
        token1 = EnhancedToken("cat", 0, "NOUN", TokenRole.PROTAGONIST, salience=0.95)
        token2 = EnhancedToken("walking", 1, "VERB", TokenRole.ACTION, salience=0.90)
        tokens = [token1, token2]
        
        # Check relationship
        rel = tokenizer._check_relationship(token1, token2)
        assert rel == RelationType.SUBJECT_VERB
    
    def test_dependency_structure(self, tokenizer):
        """Test dependency structure computation."""
        tokens = [
            EnhancedToken("cat", 0, "NOUN", TokenRole.PROTAGONIST),
            EnhancedToken("walking", 1, "VERB", TokenRole.ACTION),
            EnhancedToken("quickly", 2, "ADV", TokenRole.MODIFIER),
        ]
        
        relationships = [
            SemanticRelationship(0, 1, RelationType.SUBJECT_VERB),
            SemanticRelationship(1, 2, RelationType.ADVERB_VERB),
        ]
        
        deps = tokenizer.compute_dependency_structure(tokens, relationships)
        
        assert 0 in deps
        assert 1 in deps[0]  # Cat -> walking
    
    def test_noun_phrase_extraction(self, tokenizer):
        """Test noun phrase extraction."""
        tokens = [
            EnhancedToken("bright", 0, "ADJ", TokenRole.MODIFIER),
            EnhancedToken("cat", 1, "NOUN", TokenRole.PROTAGONIST),
            EnhancedToken("walking", 2, "VERB", TokenRole.ACTION),
        ]
        
        phrases = tokenizer.extract_noun_phrases(tokens)
        
        assert len(phrases) > 0
        # Should extract "bright cat"
        assert any("bright" in phrase[2] for phrase in phrases)
    
    def test_verb_phrase_extraction(self, tokenizer):
        """Test verb phrase extraction."""
        tokens = [
            EnhancedToken("cat", 0, "NOUN", TokenRole.PROTAGONIST),
            EnhancedToken("walking", 1, "VERB", TokenRole.ACTION),
            EnhancedToken("quickly", 2, "ADV", TokenRole.MODIFIER),
        ]
        
        phrases = tokenizer.extract_verb_phrases(tokens)
        
        assert len(phrases) > 0
    
    def test_semantic_structure(self, tokenizer):
        """Test complete semantic structure."""
        tokens = [
            EnhancedToken("cat", 0, "NOUN", TokenRole.PROTAGONIST, salience=0.95),
            EnhancedToken("walking", 1, "VERB", TokenRole.ACTION, salience=0.90),
        ]
        
        relationships = [
            SemanticRelationship(0, 1, RelationType.SUBJECT_VERB),
        ]
        
        structure = tokenizer.get_semantic_structure(tokens, relationships)
        
        assert "tokens" in structure
        assert "relationships" in structure
        assert "dependencies" in structure
        assert "salience_order" in structure


class TestSemanticIntegration:
    """Integration tests for semantic tokenization."""
    
    def test_full_tokenization_workflow(self):
        """Test complete tokenization workflow."""
        tokenizer = SemanticTokenizer()
        
        tokens = ["the", "cat", "walking", "through", "forest"]
        
        enhanced, relationships = tokenizer.tokenize_semantic(
            "the cat walking through forest",
            tokens
        )
        
        assert len(enhanced) == len(tokens)
        
        # Check role assignments
        roles = {t.text: t.role for t in enhanced}
        assert roles["cat"] == TokenRole.PROTAGONIST
        assert roles["walking"] == TokenRole.ACTION
        assert roles["forest"] == TokenRole.SETTING
    
    def test_complex_sentence(self):
        """Test tokenization of complex sentence."""
        tokenizer = SemanticTokenizer()
        
        tokens = [
            "a",
            "bright",
            "cat",
            "walking",
            "slowly",
            "in",
            "the",
            "forest",
        ]
        
        enhanced, relationships = tokenizer.tokenize_semantic(
            "a bright cat walking slowly in the forest",
            tokens
        )
        
        structure = tokenizer.get_semantic_structure(enhanced, relationships)
        
        # Check structure
        assert len(structure["noun_phrases"]) > 0
        assert len(structure["verb_phrases"]) > 0

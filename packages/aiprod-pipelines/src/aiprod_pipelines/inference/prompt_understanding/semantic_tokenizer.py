"""Advanced semantic tokenization with relationship tracking.

Provides:
- Fine-grained token analysis
- Relationship identification between tokens
- Dependency structure preservation
- Semantic role labeling
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of semantic relationships between tokens."""
    
    SUBJECT_VERB = "subject_verb"  # Subject performs action
    VERB_OBJECT = "verb_object"    # Verb acts on object
    ADJECTIVE_NOUN = "adjective_noun"  # Adjective modifies noun
    ADVERB_VERB = "adverb_verb"    # Adverb modifies action
    LOCATION = "location"          # Location/spatial relationship
    TEMPORAL = "temporal"          # Temporal relationship
    POSSESSION = "possession"      # Possession/belonging
    COMPARISON = "comparison"      # Comparative relationship
    CAUSALITY = "causality"        # Cause-effect relationship


@dataclass
class SemanticRelationship:
    """Relationship between two tokens."""
    
    source_idx: int         # First token index
    target_idx: int         # Second token index
    rel_type: RelationType  # Relationship type
    strength: float = 0.8   # 0-1 confidence
    
    def __repr__(self) -> str:
        return f"Rel({self.source_idx}->{self.target_idx}: {self.rel_type.value})"


class TokenRole(Enum):
    """Semantic roles for tokens in generation."""
    
    PROTAGONIST = "protagonist"    # Main subject
    ACTION = "action"              # Main verb/action
    ANTAGONIST = "antagonist"      # Secondary/opposing subject
    SETTING = "setting"            # Location/context
    ATTRIBUTE = "attribute"        # Quality/characteristic
    MODIFIER = "modifier"          # Adjective/adverb
    CONNECTOR = "connector"        # Preposition/conjunction


@dataclass
class EnhancedToken:
    """Token with semantic role and relationship information."""
    
    text: str                       # Original text
    idx: int                        # Position
    token_type: str                 # Part of speech
    role: TokenRole                 # Semantic role
    salience: float = 0.5           # 0-1 importance in context
    related_to: List[int] = None    # Indices of related tokens
    
    def __post_init__(self):
        if self.related_to is None:
            self.related_to = []


class SemanticTokenizer:
    """Advanced tokenization with semantic understanding.
    
    Provides:
    - Role assignment (protagonist, action, setting, etc.)
    - Relationship identification
    - Salience/importance scoring
    - Semantic dependency structures
    """
    
    def __init__(self):
        """Initialize semantic tokenizer."""
        self.role_patterns = {
            "protagonist": {"person", "character", "creature", "cat", "dog", "human", "woman", "man"},
            "action": {"walking", "running", "jumping", "flying", "moving", "dancing"},
            "setting": {"forest", "beach", "city", "mountain", "room", "village", "field"},
        }
        
        self.relationship_indicators = {
            RelationType.SUBJECT_VERB: {"performing", "doing", "making"},
            RelationType.ADJECTIVE_NOUN: {"is", "are", "being"},
            RelationType.LOCATION: {"in", "on", "at", "inside", "outside", "through"},
            RelationType.TEMPORAL: {"while", "during", "as", "when", "before", "after"},
        }
    
    def tokenize_semantic(
        self,
        text: str,
        basic_tokens: List[str],
    ) -> Tuple[List[EnhancedToken], List[SemanticRelationship]]:
        """
        Perform semantic tokenization.
        
        Args:
            text: Original prompt text (unused, for context)
            basic_tokens: Basic tokens from tokenizer
            
        Returns:
            Tuple of (enhanced tokens, relationships)
        """
        # Create enhanced tokens
        enhanced_tokens = []
        
        for idx, token in enumerate(basic_tokens):
            role = self._assign_role(token)
            salience = self._compute_salience(token, role)
            
            etoken = EnhancedToken(
                text=token.lower().strip('.,!?;:'),
                idx=idx,
                token_type=self._get_pos_tag(token),
                role=role,
                salience=salience,
            )
            enhanced_tokens.append(etoken)
        
        # Identify relationships
        relationships = self._identify_relationships(enhanced_tokens)
        
        # Update related_to lists
        for rel in relationships:
            enhanced_tokens[rel.source_idx].related_to.append(rel.target_idx)
            enhanced_tokens[rel.target_idx].related_to.append(rel.source_idx)
        
        return enhanced_tokens, relationships
    
    def _assign_role(self, token: str) -> TokenRole:
        """Assign semantic role to token."""
        token_lower = token.lower()
        
        if token_lower in self.role_patterns["protagonist"]:
            return TokenRole.PROTAGONIST
        elif token_lower in self.role_patterns["action"]:
            return TokenRole.ACTION
        elif token_lower in self.role_patterns["setting"]:
            return TokenRole.SETTING
        elif token_lower in {"very", "extremely", "incredibly"}:
            return TokenRole.MODIFIER
        elif token_lower in {"in", "on", "at", "with", "by"}:
            return TokenRole.CONNECTOR
        else:
            return TokenRole.ATTRIBUTE
    
    def _get_pos_tag(self, token: str) -> str:
        """Simplified POS tagging."""
        token_lower = token.lower()
        
        action_verbs = {"walking", "running", "jumping", "flying", "moving"}
        if token_lower in action_verbs:
            return "VERB"
        
        if token_lower in {"very", "extremely", "slowly", "quickly"}:
            return "ADV"
        
        if len(token) > 0 and token[0].isupper():
            return "NOUN"
        
        return "OTHER"
    
    def _compute_salience(self, token: str, role: TokenRole) -> float:
        """Compute token salience (importance).
        
        Returns:
            0-1 salience score
        """
        base_salience = {
            TokenRole.PROTAGONIST: 0.95,
            TokenRole.ACTION: 0.90,
            TokenRole.ANTAGONIST: 0.70,
            TokenRole.SETTING: 0.70,
            TokenRole.ATTRIBUTE: 0.50,
            TokenRole.MODIFIER: 0.40,
            TokenRole.CONNECTOR: 0.20,
        }
        
        return base_salience.get(role, 0.5)
    
    def _identify_relationships(
        self,
        tokens: List[EnhancedToken],
    ) -> List[SemanticRelationship]:
        """Identify semantic relationships between tokens.
        
        Args:
            tokens: Enhanced tokens
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for i, token1 in enumerate(tokens):
            for j, token2 in enumerate(tokens[i+1:], start=i+1):
                rel = self._check_relationship(token1, token2)
                if rel:
                    relationships.append(
                        SemanticRelationship(
                            source_idx=i,
                            target_idx=j,
                            rel_type=rel,
                            strength=0.8,
                        )
                    )
        
        return relationships
    
    def _check_relationship(
        self,
        token1: EnhancedToken,
        token2: EnhancedToken,
    ) -> Optional[RelationType]:
        """Check for relationship between two tokens."""
        # Subject-verb relationship
        if token1.role == TokenRole.PROTAGONIST and token2.role == TokenRole.ACTION:
            return RelationType.SUBJECT_VERB
        
        # Adjective-noun
        if token1.role == TokenRole.ATTRIBUTE and token2.role == TokenRole.PROTAGONIST:
            return RelationType.ADJECTIVE_NOUN
        
        # Location
        if token1.role == TokenRole.CONNECTOR and token2.role == TokenRole.SETTING:
            return RelationType.LOCATION
        
        return None
    
    def compute_dependency_structure(
        self,
        tokens: List[EnhancedToken],
        relationships: List[SemanticRelationship],
    ) -> Dict[int, List[int]]:
        """Create dependency structure (token dependencies).
        
        Args:
            tokens: Enhanced tokens
            relationships: Identified relationships
            
        Returns:
            Mapping of token index to dependent indices
        """
        dependencies = {i: [] for i in range(len(tokens))}
        
        for rel in relationships:
            # Add dependency from source to target
            dependencies[rel.source_idx].append(rel.target_idx)
        
        return dependencies
    
    def extract_noun_phrases(
        self,
        tokens: List[EnhancedToken],
    ) -> List[Tuple[int, int, str]]:
        """Extract noun phrases from token sequence.
        
        Args:
            tokens: Enhanced tokens
            
        Returns:
            List of (start_idx, end_idx, phrase_text)
        """
        phrases = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Start of potential noun phrase
            if token.role in {TokenRole.PROTAGONIST, TokenRole.SETTING}:
                start = i
                phrase_tokens = [token.text]
                
                # Look for modifiers before noun
                j = i - 1
                while j >= 0 and tokens[j].role == TokenRole.MODIFIER:
                    phrase_tokens.insert(0, tokens[j].text)
                    start = j
                    j -= 1
                
                # Look for modifiers after noun
                j = i + 1
                while j < len(tokens) and tokens[j].role == TokenRole.MODIFIER:
                    phrase_tokens.append(tokens[j].text)
                    j += 1
                
                phrase_text = " ".join(phrase_tokens)
                phrases.append((start, j, phrase_text))
                i = j
            else:
                i += 1
        
        return phrases
    
    def extract_verb_phrases(
        self,
        tokens: List[EnhancedToken],
    ) -> List[Tuple[int, int, str]]:
        """Extract verb phrases.
        
        Args:
            tokens: Enhanced tokens
            
        Returns:
            List of (start_idx, end_idx, phrase_text)
        """
        phrases = []
        
        for i, token in enumerate(tokens):
            if token.role == TokenRole.ACTION:
                start = max(0, i - 1)  # Include potential adverb
                end = min(len(tokens), i + 2)  # Include potential objects
                
                phrase_tokens = [t.text for t in tokens[start:end]]
                phrase_text = " ".join(phrase_tokens)
                
                phrases.append((start, end, phrase_text))
        
        return phrases
    
    def get_semantic_structure(
        self,
        tokens: List[EnhancedToken],
        relationships: List[SemanticRelationship],
    ) -> Dict[str, any]:
        """Get complete semantic structure.
        
        Args:
            tokens: Enhanced tokens
            relationships: Token relationships
            
        Returns:
            Dictionary with structure information
        """
        noun_phrases = self.extract_noun_phrases(tokens)
        verb_phrases = self.extract_verb_phrases(tokens)
        dependencies = self.compute_dependency_structure(tokens, relationships)
        
        return {
            "tokens": tokens,
            "relationships": relationships,
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases,
            "dependencies": dependencies,
            "salience_order": sorted(
                enumerate(tokens),
                key=lambda x: x[1].salience,
                reverse=True,
            ),
        }

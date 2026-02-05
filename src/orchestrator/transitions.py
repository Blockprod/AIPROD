"""
Transitions module pour l'orchestrateur AIPROD
Gère les transitions conditionnelles entre états selon le JSON
"""
from enum import Enum
from typing import Dict, Any
from src.utils.monitoring import logger

class TransitionCondition:
    """Évalue les conditions de transition selon le JSON AIPROD."""
    
    @staticmethod
    def evaluate(state: str, memory: Dict[str, Any]) -> str:
        """
        Évalue la prochaine transition selon l'état actuel et la mémoire.
        
        Args:
            state (str): État actuel
            memory (Dict[str, Any]): Mémoire du pipeline
            
        Returns:
            str: Prochain état
        """
        complexity_score = memory.get('complexity_score', 0.5)
        pipeline_mode = memory.get('pipeline_mode', 'full')
        technical_valid = memory.get('technical_valid', False)
        
        transitions = {
            "INIT": "ANALYSIS",
            "ANALYSIS": "FAST_TRACK" if complexity_score < 0.3 else "CREATIVE_DIRECTION",
            "CREATIVE_DIRECTION": "VISUAL_TRANSLATION",
            "VISUAL_TRANSLATION": "FINANCIAL_OPTIMIZATION",
            "FINANCIAL_OPTIMIZATION": "RENDER_EXECUTION",
            "RENDER_EXECUTION": "QA_TECHNICAL",
            "QA_TECHNICAL": "QA_SEMANTIC" if technical_valid else "ERROR",
            "QA_SEMANTIC": "FINALIZE",
            "FAST_TRACK": "FINANCIAL_OPTIMIZATION",
            "ERROR": "ANALYSIS",
            "FINALIZE": "DELIVERED"
        }
        
        next_state = transitions.get(state, "ERROR")
        logger.info(f"Transition: {state} -> {next_state} (complexity={complexity_score}, mode={pipeline_mode})")
        return next_state

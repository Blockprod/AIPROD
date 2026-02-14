"""
A/B Testing Framework for Reward Model Evaluation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""
    test_id: str
    variant_a: Dict  # Hyperparameters for variant A
    variant_b: Dict  # Hyperparameters for variant B
    sample_size_per_variant: int = 100
    significance_level: float = 0.05  # p-value threshold


@dataclass
class ABTestResult:
    """Result of single A/B test"""
    test_id: str
    variant_a_scores: List[float] = field(default_factory=list)
    variant_b_scores: List[float] = field(default_factory=list)
    
    def summary(self) -> Dict:
        """Get statistical summary"""
        a_scores = np.array(self.variant_a_scores)
        b_scores = np.array(self.variant_b_scores)
        
        return {
            "test_id": self.test_id,
            "variant_a": {
                "mean": float(np.mean(a_scores)) if len(a_scores) > 0 else 0.0,
                "std": float(np.std(a_scores)) if len(a_scores) > 0 else 0.0,
                "sample_size": len(a_scores),
            },
            "variant_b": {
                "mean": float(np.mean(b_scores)) if len(b_scores) > 0 else 0.0,
                "std": float(np.std(b_scores)) if len(b_scores) > 0 else 0.0,
                "sample_size": len(b_scores),
            },
            "winner": self._determine_winner(),
        }
    
    def _determine_winner(self) -> Optional[str]:
        """Determine statistical winner"""
        if not self.variant_a_scores or not self.variant_b_scores:
            return None
        
        a_mean = np.mean(self.variant_a_scores)
        b_mean = np.mean(self.variant_b_scores)
        
        if a_mean > b_mean:
            return "A"
        elif b_mean > a_mean:
            return "B"
        else:
            return "TIE"


class ABTestingFramework:
    """Framework for A/B testing hyperparameter configurations"""
    
    def __init__(self):
        """Initialize framework"""
        self.active_tests: Dict[str, ABTestResult] = {}
        self.completed_tests: List[ABTestResult] = []
    
    async def start_ab_test(
        self,
        test_id: str,
        variant_a: Dict,
        variant_b: Dict,
        sample_size: int = 100,
    ) -> ABTestConfig:
        """
        Start new A/B test.
        
        Args:
            test_id: Unique test identifier
            variant_a: Dict of hyperparameters for variant A
            variant_b: Dict of hyperparameters for variant B
            sample_size: Samples per variant
            
        Returns:
            ABTestConfig
        """
        config = ABTestConfig(
            test_id=test_id,
            variant_a=variant_a,
            variant_b=variant_b,
            sample_size_per_variant=sample_size,
        )
        
        self.active_tests[test_id] = ABTestResult(test_id=test_id)
        
        return config
    
    async def record_result(
        self,
        test_id: str,
        variant: str,
        score: float,
    ):
        """
        Record result for variant.
        
        Args:
            test_id: Test ID
            variant: "A" or "B"
            score: Quality score 0-1
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")
        
        test = self.active_tests[test_id]
        
        if variant.upper() == "A":
            test.variant_a_scores.append(score)
        elif variant.upper() == "B":
            test.variant_b_scores.append(score)
        else:
            raise ValueError(f"Invalid variant: {variant}")
    
    async def get_test_summary(self, test_id: str) -> Dict:
        """Get current test summary"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")
        
        return self.active_tests[test_id].summary()
    
    async def complete_test(self, test_id: str, winner: Optional[str] = None) -> Dict:
        """
        Mark test as complete.
        
        Args:
            test_id: Test ID
            winner: Force winner (e.g., "A", "B", or None for auto-determine)
            
        Returns:
            Final test summary
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test not found: {test_id}")
        
        test = self.active_tests.pop(test_id)
        self.completed_tests.append(test)
        
        summary = test.summary()
        if winner:
            summary["forced_winner"] = winner
        
        return summary
    
    def get_test_history(self) -> List[Dict]:
        """Get history of all completed tests"""
        return [test.summary() for test in self.completed_tests]

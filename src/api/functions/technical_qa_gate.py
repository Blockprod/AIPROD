"""
TechnicalQAGate pour AIPROD V33
Vérifications binaires déterministes pour la validation technique.
"""
from typing import Any, Dict, List
from src.utils.monitoring import logger

class TechnicalQAGate:
    """
    Gate de validation technique avec vérifications binaires déterministes.
    Pas de LLM, règles métier pures.
    """
    def __init__(self):
        self.checks = {
            "asset_count": lambda m: len(m.get("assets", [])) > 0,
            "manifest_complete": lambda m: all(k in m for k in ["complexity_score", "assets"]),
            "cost_valid": lambda m: m.get("cost", 0) >= 0,
            "quality_acceptable": lambda m: m.get("quality_score", 0) >= 0.5
        }

    def validate(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide le manifeste selon les règles techniques.
        Args:
            manifest (Dict[str, Any]): Manifeste à valider.
        Returns:
            Dict[str, Any]: Rapport de validation avec détails des checks.
        """
        logger.info("TechnicalQAGate: start validation")
        results = {}
        for check_name, check_func in self.checks.items():
            try:
                results[check_name] = check_func(manifest)
            except Exception as e:
                logger.error(f"TechnicalQAGate: check {check_name} failed {e}")
                results[check_name] = False
        
        passed = all(results.values())
        report = {
            "technical_valid": passed,
            "checks": results,
            "details": "All checks passed" if passed else "Some checks failed"
        }
        logger.info(f"TechnicalQAGate: report={report}")
        return report

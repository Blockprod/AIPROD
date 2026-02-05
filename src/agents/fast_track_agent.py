"""
FastTrackAgent pour AIPROD
Gère le pipeline simplifié et les contraintes de rapidité.
"""
import asyncio
from typing import Any, Dict
from src.utils.monitoring import logger

class FastTrackAgent:
    """
    Agent pour le pipeline Fast Track (exécution rapide, contraintes minimales).
    """
    def __init__(self):
        pass

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute le pipeline simplifié avec contraintes de rapidité.

        Args:
            inputs (Dict[str, Any]): Données d'entrée.
        Returns:
            Dict[str, Any]: Output Fast Track.
        """
        logger.info("FastTrackAgent: start fast pipeline")
        # Mock : vérifie une contrainte simple
        content = inputs.get("content", "")
        priority = inputs.get("priority", "normal")
        if priority == "high":
            await asyncio.sleep(0.1)  # Simule une exécution rapide
            output = {"status": "fast_tracked", "inputs": {**inputs, "text_prompt": content}}
        else:
            await asyncio.sleep(0.5)
            output = {"status": "normal", "inputs": {**inputs, "text_prompt": content}}
        logger.info(f"FastTrackAgent: output={output}")
        return output

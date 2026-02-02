"""
LLM Wrappers pour AIPROD V33
Abstraction pour les appels LLM (Gemini, fallbacks)
Basé sur les configurations du JSON AIPROD_V33.json
"""
from typing import Any, Dict, List, Optional
import asyncio
from src.utils.monitoring import logger

class LLMWrapper:
    """
    Wrapper générique pour les appels LLM avec fallback.
    Supporte Gemini et autres providers.
    """
    
    def __init__(self, provider: str = "google", model: str = "gemini-1.5-pro"):
        self.provider = provider
        self.model = model
        self.fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
        self.timeout_sec = 60
        self.max_tokens = 8000
        logger.info(f"LLMWrapper initialized: provider={provider}, model={model}")
    
    async def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Appelle le LLM avec fallback automatique.
        
        Args:
            prompt (str): Prompt utilisateur
            system_prompt (Optional[str]): System prompt
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict[str, Any]: Réponse du LLM
        """
        logger.info(f"LLMWrapper: calling {self.model}")
        try:
            # Mock: simule un appel LLM
            await asyncio.sleep(0.1)
            return {
                "response": f"Mock response from {self.model}",
                "model": self.model,
                "usage": {"tokens": 150, "cost": 0.001},
                "success": True
            }
        except Exception as e:
            logger.error(f"LLMWrapper: error with {self.model}: {e}")
            return await self._fallback(prompt, system_prompt, **kwargs)
    
    async def _fallback(self, prompt: str, system_prompt: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Tente les modèles de fallback.
        """
        for fallback_model in self.fallback_models:
            logger.info(f"LLMWrapper: trying fallback {fallback_model}")
            try:
                await asyncio.sleep(0.1)
                return {
                    "response": f"Fallback response from {fallback_model}",
                    "model": fallback_model,
                    "usage": {"tokens": 100, "cost": 0.0005},
                    "success": True,
                    "fallback": True
                }
            except Exception as e:
                logger.error(f"LLMWrapper: fallback {fallback_model} failed: {e}")
        
        return {"success": False, "error": "All models failed"}
    
    def set_timeout(self, timeout_sec: int) -> None:
        """Configure le timeout."""
        self.timeout_sec = timeout_sec
    
    def set_max_tokens(self, max_tokens: int) -> None:
        """Configure le nombre max de tokens."""
        self.max_tokens = max_tokens


class GeminiWrapper(LLMWrapper):
    """
    Wrapper spécialisé pour Gemini avec configuration JSON.
    """
    
    def __init__(self, model: str = "gemini-1.5-pro"):
        super().__init__(provider="google", model=model)
        self.vision_model = "gemini-1.5-pro-vision"
    
    async def call_vision(self, image_url: str, prompt: str) -> Dict[str, Any]:
        """
        Appelle Gemini Vision pour l'analyse d'image.
        """
        logger.info(f"GeminiWrapper: calling vision model")
        await asyncio.sleep(0.15)
        return {
            "response": f"Vision analysis of {image_url}",
            "model": self.vision_model,
            "analysis": {
                "objects": ["person", "landscape"],
                "quality_score": 0.85,
                "style": "cinematic"
            },
            "success": True
        }

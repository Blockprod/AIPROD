"""
CreativeDirector Agent pour AIPROD
Génère des scripts vidéo via Gemini.
Utilise le consistency cache pour maintenir la cohérence marque.
"""
import asyncio
import os
from typing import Any, Dict, Optional
import httpx
from src.utils.cache_manager import CacheManager
from src.utils.monitoring import logger
from src.memory.consistency_cache import get_consistency_cache


class CreativeDirector:
    """
    Agent principal pour générer des scripts via Gemini.
    Intègre le cache de cohérence pour la réutilisation des marqueurs de style.
    """
    def __init__(self):
        self.cache = CacheManager(ttl_hours=168)
        self.consistency_cache = get_consistency_cache()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un script vidéo via Gemini.
        Utilise le consistency cache si brand_id est fourni.

        Args:
            inputs (Dict[str, Any]): Données d'entrée (content, lang, brand_id, etc.).
        Returns:
            Dict[str, Any]: Script et manifest avec consistency_markers.
        """
        logger.info("CreativeDirector: start script generation")
        
        content = inputs.get("content", "")
        brand_id = inputs.get("brand_id", "default")
        use_consistency_cache = inputs.get("_preset_config", {}).get("consistency_cache", False)
        
        # Check consistency cache si activé (preset brand_campaign ou premium_spot)
        cached_markers = None
        if use_consistency_cache:
            cached_markers = self.consistency_cache.get_consistency_markers(
                brand_id=brand_id,
                content=content
            )
            if cached_markers:
                logger.info(f"CreativeDirector: consistency cache HIT for brand={brand_id}")
        
        # Check script cache
        cache_key = f"script_{hash(content)}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("CreativeDirector: script cache hit")
            # Enrichir avec les marqueurs de cohérence si disponibles
            if cached_markers:
                cached["consistency_markers"] = cached_markers
                cached["consistency_cache_hit"] = True
            return cached
        
        # Générer via Gemini
        if not self.gemini_api_key or self.gemini_api_key == "your-gemini-api-key":
            logger.warning("CreativeDirector: No Gemini key, using mock")
            output = self._mock_output(inputs, cached_markers)
        else:
            output = await self._generate_with_gemini(inputs, cached_markers)
        
        # Stocker les nouveaux marqueurs de cohérence si activé
        if use_consistency_cache and not cached_markers:
            new_markers = output.get("consistency_markers", {})
            if new_markers:
                self.consistency_cache.set_consistency_markers(
                    brand_id=brand_id,
                    content=content,
                    markers=new_markers
                )
                logger.info(f"CreativeDirector: consistency markers STORED for brand={brand_id}")
        
        # Cache le résultat
        self.cache.set(cache_key, output)
        logger.info("CreativeDirector: script generation complete")
        return output
    
    def _mock_output(self, inputs: Dict[str, Any], cached_markers: Optional[Dict] = None) -> Dict[str, Any]:
        """Fallback mock si pas de clé Gemini."""
        script = f"Script vidéo pour: {inputs.get('content')}"
        
        # Utiliser les marqueurs cachés ou en générer de nouveaux
        consistency_markers = cached_markers or {
            "style": "modern",
            "color_palette": ["#FF5733", "#33FF57", "#3357FF"],
            "mood": "dynamic",
            "lighting": "bright",
            "character_style": "realistic"
        }
        
        return {
            "script": script,
            "production_manifest": {
                "scenes": ["Scene 1", "Scene 2"],
                "duration": 5
            },
            "complexity_score": 0.5,
            "inputs": {"text_prompt": script},
            "consistency_markers": consistency_markers,
            "consistency_cache_hit": cached_markers is not None
        }

    async def _generate_with_gemini(self, inputs: Dict[str, Any], cached_markers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Génère un script via l'API Gemini.
        Intègre les marqueurs de cohérence si disponibles.
        """
        content = inputs.get("content", "")
        lang = inputs.get("lang", "en")
        
        # Ajouter les marqueurs de cohérence au prompt si disponibles
        consistency_context = ""
        if cached_markers:
            consistency_context = f"""
Utilise ces marqueurs de cohérence pour maintenir le style de la marque:
- Style: {cached_markers.get('style', 'modern')}
- Palette de couleurs: {cached_markers.get('color_palette', [])}
- Ambiance: {cached_markers.get('mood', 'dynamic')}
- Éclairage: {cached_markers.get('lighting', 'bright')}
"""
        
        prompt = f"""Tu es un expert en scénarios vidéo.
Génère un script de vidéo de 5 secondes pour: "{content}"
Langue: {lang}
{consistency_context}

Réponds au format JSON:
{{
  "script": "description détaillée scène par scène",
  "scenes": ["scène 1", "scène 2"],
  "duration": 5,
  "complexity_score": 0.0-1.0,
  "consistency_markers": {{
    "style": "modern/classic/minimal",
    "color_palette": ["#hex1", "#hex2"],
    "mood": "dynamic/calm/dramatic",
    "lighting": "bright/dark/natural",
    "character_style": "realistic/cartoon/abstract"
  }}
}}"""
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2048
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Extraire le texte de réponse
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parser JSON (basique)
            import json
            try:
                result = json.loads(text.strip("```json\n").strip("\n```"))
            except:
                result = self._mock_output(inputs, cached_markers)
                return result
            
            # Utiliser les marqueurs cachés si disponibles, sinon ceux générés
            consistency_markers = cached_markers or result.get("consistency_markers", {})
            
            return {
                "script": result.get("script", ""),
                "production_manifest": {
                    "scenes": result.get("scenes", []),
                    "duration": result.get("duration", 5)
                },
                "complexity_score": result.get("complexity_score", 0.5),
                "inputs": {"text_prompt": result.get("script", "")},
                "consistency_markers": consistency_markers,
                "consistency_cache_hit": cached_markers is not None
            }

    async def fallback_gemini(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback Gemini.
        """
        return await self._generate_with_gemini(inputs)
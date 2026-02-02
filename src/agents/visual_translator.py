"""
VisualTranslator Agent pour AIPROD V33
Traduit et adapte les assets visuels avec Gemini.
"""

import asyncio
import os
import json
from typing import Any, Dict, Optional
from google import genai
from src.utils.monitoring import logger


class VisualTranslator:
    """
    Agent responsable de la traduction/adaptation des assets visuels.
    Utilise Gemini pour générer des instructions d'adaptation visuelle.
    """

    def __init__(self):
        """Initialise le client Gemini."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key and api_key != "your-gemini-api-key":
            self.client = genai.Client(api_key=api_key)
            self.use_real_gemini = True
        else:
            self.client = None
            self.use_real_gemini = False
            logger.warning(
                "VisualTranslator: No Gemini API key, using mock translation"
            )

    async def run(
        self, assets: Dict[str, Any], target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Traduit/adapte les assets visuels avec Gemini.

        Args:
            assets (Dict[str, Any]): Assets à traduire/adapter.
            target_lang (str): Langue cible (ex: "en", "fr", "es", "de").
        Returns:
            Dict[str, Any]: Assets traduits/adaptés avec instructions d'adaptation.
        """
        logger.info(f"VisualTranslator: start translation to {target_lang}")

        if not self.use_real_gemini or not self.client:
            return self._mock_translation(assets, target_lang)

        try:
            # Préparer les données pour la traduction
            assets_summary = json.dumps(assets, indent=2, default=str)[:2000]

            # Construire le prompt de traduction visuelle
            translation_prompt = f"""Vous êtes un expert en adaptation visuelle et culturelle de contenu multimédia.

Vous devez adapter les assets visuels suivants pour une audience parlant {target_lang.upper()}:

**Assets actuels:**
{assets_summary}

Pour chaque asset, fournissez:
1. **Traduction du texte** (le cas échéant)
2. **Adaptations culturelles** (couleurs, symboles, conventions locales)
3. **Instructions de redesign** (typographie, layout, emojis adaptés)
4. **Notes de localisation** (considérations régionales)

Format de réponse JSON:
{{
  "adapted_assets": {{
    "asset_name": {{
      "translated_text": "texte traduit",
      "cultural_adaptations": ["adaptation1", "adaptation2"],
      "design_instructions": "instructions détaillées",
      "localization_notes": "notes importantes"
    }}
  }},
  "language_code": "xx",
  "cultural_insights": ["insight1", "insight2"],
  "readiness_score": 0.0-1.0,
  "status": "adapted"
}}"""

            # Appel à Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp", contents=translation_prompt
            )
            response_text = response.text if response.text else ""

            # Parser la réponse JSON
            try:
                json_match = response_text.find("{")
                if json_match != -1:
                    json_text = response_text[json_match:]
                    json_end = json_text.rfind("}") + 1
                    result = json.loads(json_text[:json_end])
                else:
                    result = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError:
                logger.warning(f"VisualTranslator: Failed to parse JSON response")
                return self._mock_translation(assets, target_lang)

            # Construire la réponse finale
            report = {
                "status": "adapted",
                "language": target_lang,
                "adapted_assets": result.get("adapted_assets", assets),
                "cultural_insights": result.get("cultural_insights", []),
                "readiness_score": result.get("readiness_score", 0.8),
                "localization_metadata": {
                    "target_language": target_lang,
                    "adapted_by": "gemini",
                    "adaptation_timestamp": __import__("datetime")
                    .datetime.now(__import__("datetime").timezone.utc)
                    .isoformat(),
                },
                "provider": "gemini",
            }

            logger.info(f"VisualTranslator: translation complete (lang={target_lang})")
            return report

        except Exception as e:
            logger.error(f"VisualTranslator: Gemini API error: {e}")
            return self._mock_translation(assets, target_lang)

    def _mock_translation(
        self, assets: Dict[str, Any], target_lang: str = "en"
    ) -> Dict[str, Any]:
        """Fallback mock translation."""
        logger.info(f"VisualTranslator: using mock translation for {target_lang}")
        adapted = {k: f"{v}_translated_{target_lang}" for k, v in assets.items()}
        return {
            "status": "adapted",
            "language": target_lang,
            "adapted_assets": adapted,
            "cultural_insights": ["Mock translation applied"],
            "readiness_score": 0.7,
            "localization_metadata": {
                "target_language": target_lang,
                "adapted_by": "mock",
                "adaptation_timestamp": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(),
            },
            "provider": "mock",
        }

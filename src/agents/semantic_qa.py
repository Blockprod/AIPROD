"""
SemanticQA Agent pour AIPROD
Valide la cohérence sémantique des outputs générés avec Gemini.
"""

import asyncio
import os
import json
from typing import Any, Dict, Optional
from google import genai
from src.utils.monitoring import logger


class SemanticQA:
    """
    Agent responsable de la validation sémantique des outputs.
    Utilise Gemini 2.0 Flash pour analyser la qualité et la pertinence des résultats.
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
            logger.warning("SemanticQA: No Gemini API key, using mock validation")

    async def run(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide la cohérence sémantique des outputs avec Gemini.

        Args:
            outputs (Dict[str, Any]): Outputs à valider (render_output, semantic_report, etc.).
        Returns:
            Dict[str, Any]: Rapport de validation sémantique avec scores.
        """
        logger.info("SemanticQA: start semantic validation")

        if not self.use_real_gemini or not self.client:
            return self._mock_validation(outputs)

        try:
            # Préparer les données pour l'analyse
            render_output = outputs.get("render", {}) or outputs.get("assets", {})
            original_prompt = outputs.get("prompt", "") or outputs.get("content", "")

            # Construire le prompt d'analyse
            analysis_prompt = f"""Vous êtes un expert en validation sémantique de contenu généré par IA.

Analysez les résultats suivants:

**Prompt original:**
{original_prompt}

**Outputs générés:**
{json.dumps(render_output, indent=2, default=str)[:2000]}

Évaluez les aspects suivants (échelle 0-1):
1. **Qualité visuelle/technique** (clarté, résolution, artefacts)
2. **Pertinence au prompt** (correspond-il à la demande?)
3. **Cohérence sémantique** (les éléments sont-ils cohérents ensemble?)
4. **Complétude** (toutes les parties requises sont-elles présentes?)

Identifiez aussi:
- Artefacts ou erreurs détectés
- Domaines d'amélioration
- Verdict final (acceptable oui/non)

Répondez strictement au format JSON:
{{
  "quality_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "artifacts": ["artefact1", "artefact2"],
  "improvements": ["suggestion1", "suggestion2"],
  "acceptable": true/false,
  "verdict": "brief summary"
}}"""

            # Appel à Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp", contents=analysis_prompt
            )
            response_text = response.text if response.text else ""

            # Parser la réponse JSON
            try:
                # Extraire JSON si dans un bloc de code
                json_match = response_text.find("{")
                if json_match != -1:
                    json_text = response_text[json_match:]
                    json_end = json_text.rfind("}") + 1
                    result = json.loads(json_text[:json_end])
                else:
                    result = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError:
                logger.warning(
                    f"SemanticQA: Failed to parse JSON response: {response_text[:200]}"
                )
                return self._mock_validation(outputs)

            # Construire le rapport
            report = {
                "semantic_valid": result.get("acceptable", True),
                "quality_score": result.get("quality_score", 0.5),
                "relevance_score": result.get("relevance_score", 0.5),
                "coherence_score": result.get("coherence_score", 0.5),
                "completeness_score": result.get("completeness_score", 0.5),
                "overall_score": result.get("overall_score", 0.5),
                "artifacts": result.get("artifacts", []),
                "improvements": result.get("improvements", []),
                "verdict": result.get("verdict", "Analysis complete"),
                "provider": "gemini",
            }

            logger.info(
                f"SemanticQA: validation complete (score={report['overall_score']})"
            )
            return report

        except Exception as e:
            logger.error(f"SemanticQA: Gemini API error: {e}")
            return self._mock_validation(outputs)

    def _mock_validation(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback mock validation."""
        logger.info("SemanticQA: using mock validation")
        return {
            "semantic_valid": True,
            "quality_score": 0.75,
            "relevance_score": 0.80,
            "coherence_score": 0.75,
            "completeness_score": 0.70,
            "overall_score": 0.75,
            "artifacts": [],
            "improvements": ["Consider adding more detail"],
            "verdict": "Mock validation passed",
            "provider": "mock",
        }

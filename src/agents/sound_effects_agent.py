"""
SoundEffectsAgent pour AIPROD V33
Génère des bruitages (SFX) et effets sonores contextuels pour les vidéos.
Supports: Freesound API (primary), Mock (fallback)
"""
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SoundEffectsAgent:
    """
    Génère des bruitages et effets sonores adaptés au contenu vidéo.
    
    Utilise Freesound API pour accéder à une vaste librairie de SFX.
    Fallback sur mock pour développement/testing.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or "freesound"
        self.freesound_api_key = os.getenv("FREESOUND_API_KEY")
        self.freesound_base_url = "https://freesound.org/api/v2"
        logger.info(f"SoundEffectsAgent initialized with provider: {self.provider}")

    def search_sfx_freesound(
        self,
        query: str,
        duration_min: int = 1,
        duration_max: int = 30,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recherche des bruitages sur Freesound API.
        
        Args:
            query: Description de l'effet sonore désiré
            duration_min: Durée minimale en secondes
            duration_max: Durée maximale en secondes
            max_results: Nombre de résultats à retourner
            
        Returns:
            Liste des SFX trouvés avec URLs
        """
        api_key = self.freesound_api_key
        if not api_key:
            logger.warning("FREESOUND_API_KEY not configured, using mock SFX")
            raise RuntimeError("FREESOUND_API_KEY non défini")

        try:
            # Endpoint de recherche Freesound
            search_url = f"{self.freesound_base_url}/search/text/"

            params = {
                "query": query,
                "fields": "id,name,description,previews,duration,license,download",
                "sort": "rating_desc",
                "limit": max_results,
                "duration": f"[{duration_min} TO {duration_max}]"
            }

            headers = {"Authorization": f"Token {api_key}"}

            logger.info(
                f"Freesound: Searching for SFX - query='{query}', "
                f"duration=[{duration_min}-{duration_max}]s"
            )
            
            response = requests.get(search_url, params=params, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                sfx_list = []
                for sound in results[:max_results]:
                    sfx = {
                        "id": sound.get("id"),
                        "name": sound.get("name"),
                        "description": sound.get("description", ""),
                        "preview_url": sound.get("previews", {}).get("preview-lq-mp3"),
                        "download_url": sound.get("previews", {}).get("preview-lq-mp3"),
                        "duration": sound.get("duration", 0),
                        "license": sound.get("license", "unknown"),
                        "provider": "freesound"
                    }
                    sfx_list.append(sfx)

                logger.info(f"Freesound: Found {len(sfx_list)} SFX matches for '{query}'")
                return sfx_list
            else:
                error_msg = response.text
                logger.error(f"Freesound API error {response.status_code}: {error_msg}")
                return []

        except requests.exceptions.Timeout:
            logger.warning("Freesound API timeout")
            return []
        except Exception as e:
            logger.error(f"Freesound API exception: {str(e)}")
            return []

    def generate_sfx(
        self,
        sfx_descriptions: List[str],
        duration: int = 30,
        intensity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Génère une collection d'effets sonores basée sur les descriptions.
        
        Args:
            sfx_descriptions: Liste de descriptions d'effets (ex: ["wind", "birds", "water"])
            duration: Durée totale en secondes
            intensity: Intensité sonore (low, medium, high)
            
        Returns:
            Dictionnaire avec liste d'SFX et métadonnées
        """
        try:
            # Essayer Freesound d'abord
            if self.provider == "freesound" and self.freesound_api_key:
                sfx_collection = []
                
                # Rechercher chaque description
                for sfx_desc in sfx_descriptions:
                    try:
                        matches = self.search_sfx_freesound(
                            query=sfx_desc,
                            duration_max=duration,
                            max_results=3
                        )
                        if matches:
                            sfx_collection.append(matches[0])  # Prendre le meilleur match
                    except Exception as e:
                        logger.warning(f"Failed to find SFX for '{sfx_desc}': {str(e)}")

                if sfx_collection:
                    return {
                        "sfx_list": sfx_collection,
                        "provider": "freesound",
                        "status": "completed",
                        "count": len(sfx_collection),
                        "description": f"Generated {len(sfx_collection)} SFX effects"
                    }

        except Exception as e:
            logger.error(f"SFX generation failed: {str(e)}")

        # Fallback à mock
        return self._generate_mock_sfx(sfx_descriptions, duration, intensity)

    def _generate_mock_sfx(
        self,
        sfx_descriptions: List[str],
        duration: int = 30,
        intensity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Génère des SFX mock pour développement/testing.
        """
        mock_sfx_list = []
        for desc in sfx_descriptions:
            mock_sfx_list.append({
                "id": f"mock-sfx-{desc.replace(' ', '-')}",
                "name": f"Mock {desc} effect",
                "description": f"Mock sound effect for {desc}",
                "preview_url": f"mock_sfx_{desc}.mp3",
                "duration": duration,
                "intensity": intensity,
                "provider": "mock"
            })

        return {
            "sfx_list": mock_sfx_list,
            "provider": "mock",
            "status": "completed",
            "count": len(mock_sfx_list),
            "description": f"Generated {len(mock_sfx_list)} mock SFX effects"
        }

    def extract_sfx_from_script(self, script: str) -> List[str]:
        """
        Extrait les descriptions d'effets sonores du script vidéo.
        
        Utilise des heuristiques simples ou pourrait être amélioré avec NLP/LLM.
        
        Args:
            script: Texte du script vidéo
            
        Returns:
            Liste d'descriptions d'effets sonores
        """
        # Mots-clés courants pour les SFX dans les scripts vidéo
        sfx_keywords = {
            "rain": ["pluie", "rainfall", "rainy"],
            "wind": ["vent", "wind", "windy"],
            "thunder": ["tonnerre", "thunder", "lightning"],
            "water": ["eau", "water", "splash", "river"],
            "birds": ["oiseaux", "birds", "chirping"],
            "traffic": ["trafic", "traffic", "cars", "vehicles"],
            "footsteps": ["pas", "footsteps", "walking"],
            "door": ["porte", "door", "opening"],
            "bells": ["cloche", "bells", "ringing"],
            "music": ["musique", "music", "melody"]
        }

        detected_sfx = []
        script_lower = script.lower()

        for sfx_type, keywords in sfx_keywords.items():
            for keyword in keywords:
                if keyword in script_lower:
                    if sfx_type not in detected_sfx:
                        detected_sfx.append(sfx_type)
                    break

        logger.info(f"Extracted {len(detected_sfx)} SFX types from script: {detected_sfx}")
        return detected_sfx if detected_sfx else ["ambient", "nature"]

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère les bruitages et les ajoute au manifest.
        
        Interface principale de l'agent.
        """
        script = manifest.get("script", "")
        duration = manifest.get("duration", 30)
        intensity = manifest.get("intensity", "medium")

        # Extraire descriptions d'SFX du script
        sfx_descriptions = self.extract_sfx_from_script(script)

        # Générer les SFX
        sfx_result = self.generate_sfx(
            sfx_descriptions=sfx_descriptions,
            duration=duration,
            intensity=intensity
        )

        manifest["sound_effects"] = sfx_result
        return manifest

"""
MusicComposer Agent pour AIPROD V33
Génère une bande-son contextuelle adaptée au script/scènes via API musicale.
Supports: Suno (primary), Soundful (legacy), Mock (fallback)
"""
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class MusicComposer:
    """
    Génère une musique de fond adaptée au contenu/scènes.
    Utilise Suno API si disponible, sinon Soundful, sinon fallback mock.
    Extensible pour d'autres fournisseurs.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or "suno"  # Changed default to Suno
        self.suno_api_key = os.getenv("SUNO_API_KEY")
        self.soundful_api_key = os.getenv("SOUNDFUL_API_KEY")
        logger.info(f"MusicComposer initialized with provider: {self.provider}")

    def generate_music_suno(
        self,
        style: str,
        duration: int,
        mood: Optional[str] = None,
        script: str = ""
    ) -> Dict[str, Any]:
        """
        Génère une musique via l'API Suno.
        Suno API docs: https://api.suno.ai/docs
        """
        api_key = self.suno_api_key
        if not api_key:
            logger.warning("SUNO_API_KEY not configured, falling back to mock")
            raise RuntimeError("SUNO_API_KEY non défini")

        try:
            # Suno API endpoint pour générer de la musique
            url = "https://api.suno.ai/api/generate"

            # Construire le prompt musical basé sur le style, mood et script
            prompt = self._build_music_prompt(style, mood, script)

            # Paramètres pour Suno
            payload = {
                "prompt": prompt,
                "duration": min(duration, 30),  # Suno max duration is typically 30s
                "style": style,
                "gpt_description_prompt": f"Generate background music for: {script[:100] if script else 'video content'}"
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            logger.info(
                f"Suno: Calling API to generate music - style={style}, "
                f"duration={duration}s, mood={mood}"
            )
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                music_url = data.get("music_url") or data.get("url")
                song_id = data.get("id") or data.get("song_id")

                result = {
                    "music_url": music_url,
                    "provider": "suno",
                    "style": style,
                    "duration": duration,
                    "mood": mood,
                    "song_id": song_id,
                    "status": "completed"
                }
                logger.info(f"Suno: Music generated successfully - song_id={song_id}")
                return result
            elif response.status_code == 202:
                # Suno may return 202 Accepted with a job ID for async processing
                data = response.json()
                song_id = data.get("id") or data.get("song_id")

                result = {
                    "song_id": song_id,
                    "provider": "suno",
                    "style": style,
                    "duration": duration,
                    "mood": mood,
                    "status": "pending",
                    "description": f"Musique en génération (Suno async job: {song_id})"
                }
                logger.info(f"Suno: Music generation queued - job_id={song_id}")
                return result
            else:
                error_msg = response.text
                logger.error(f"Suno API error {response.status_code}: {error_msg}")
                return {
                    "error": error_msg,
                    "provider": "suno",
                    "status": "failed"
                }
        except requests.exceptions.Timeout:
            logger.warning("Suno API timeout, falling back to mock")
            return {
                "error": "Suno API timeout",
                "provider": "suno_timeout",
                "status": "failed"
            }
        except Exception as e:
            logger.error(f"Suno API exception: {str(e)}")
            return {
                "error": str(e),
                "provider": "suno",
                "status": "error"
            }

    def generate_music_soundful(self, style: str, duration: int, mood: Optional[str] = None) -> Dict[str, Any]:
        """
        Génère une musique via Soundful API (legacy fallback).
        """
        api_key = self.soundful_api_key
        if not api_key:
            raise RuntimeError("SOUNDFUL_API_KEY non défini")

        url = "https://api.soundful.com/v1/generate"
        payload = {
            "style": style,
            "duration": duration,
            "mood": mood or "cinematic"
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                music_url = response.json().get("music_url")
                return {
                    "music_url": music_url,
                    "provider": "soundful",
                    "style": style,
                    "duration": duration,
                    "mood": mood
                }
            else:
                return {"error": response.text, "provider": "soundful"}
        except Exception as e:
            logger.error(f"Soundful API error: {str(e)}")
            return {"error": str(e), "provider": "soundful"}

    def _build_music_prompt(self, style: str, mood: Optional[str], script: str) -> str:
        """
        Construit un prompt détaillé pour Suno basé sur le style, ambiance et script.
        """
        mood_str = f", {mood}" if mood else ""
        prompt = f"Generate {style} background music{mood_str} for video content"
        if script:
            # Ajouter des détails du script si disponible (max 100 chars)
            script_summary = script[:100].strip()
            prompt += f" about: {script_summary}"
        return prompt

    def generate_music(
        self,
        script: str,
        style: str = "cinematic",
        duration: int = 30,
        mood: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère une musique adaptée au script, style et ambiance.
        Essaye Suno en premier, puis Soundful, puis fallback mock.
        """
        try:
            # Essayer Suno d'abord (nouvelle API prioritaire)
            if self.provider == "suno" or (self.provider == "auto" and self.suno_api_key):
                try:
                    result = self.generate_music_suno(style, duration, mood, script)
                    if result.get("status") != "error" and result.get("error") is None:
                        return result
                except Exception as e:
                    logger.warning(f"Suno failed: {str(e)}, trying Soundful...")

            # Fallback sur Soundful
            if self.soundful_api_key:
                return self.generate_music_soundful(style, duration, mood)

        except Exception as e:
            logger.error(f"Music generation failed: {str(e)}")

        # Fallback mock
        return {
            "music_url": f"mock_music_{style}_{duration}s.mp3",
            "provider": "mock",
            "style": style,
            "duration": duration,
            "mood": mood,
            "description": f"Musique mock pour style={style}, mood={mood}, durée={duration}s."
        }

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère la bande-son contextuelle et l'ajoute au manifest.
        """
        script = manifest.get("script", "")
        style = manifest.get("music_style", "cinematic")
        duration = manifest.get("duration", 30)
        mood = manifest.get("mood", None)
        music_result = self.generate_music(script, style=style, duration=duration, mood=mood)
        manifest["music"] = music_result
        return manifest

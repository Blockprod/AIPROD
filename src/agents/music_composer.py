"""
MusicComposer Agent pour AIPROD V33
Génère une bande-son contextuelle adaptée au script/scènes via API musicale.
"""
import os
from typing import Any, Dict, Optional

import requests

class MusicComposer:
    """
    Génère une musique de fond adaptée au contenu/scènes.
    Utilise Soundful API si disponible, sinon fallback mock.
    Extensible pour d'autres fournisseurs.
    """
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or "soundful"
        self.soundful_api_key = os.getenv("SOUNDFUL_API_KEY")

    def generate_music_soundful(self, style, duration, mood):
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

    def generate_music(self, script: str, style: str = "cinematic", duration: int = 30, mood: Optional[str] = None) -> Dict[str, Any]:
        """
        Génère une musique adaptée au script, style et ambiance.
        """
        try:
            if self.provider == "soundful" and self.soundful_api_key:
                return self.generate_music_soundful(style, duration, mood)
            # Extension future : ajouter d'autres providers ici
        except Exception as e:
            return {"error": str(e), "provider": self.provider}
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
        Génère la bande-son contextuelle et l’ajoute au manifest.
        """
        script = manifest.get("script", "")
        style = manifest.get("music_style", "cinematic")
        duration = manifest.get("duration", 30)
        mood = manifest.get("mood", None)
        music_result = self.generate_music(script, style=style, duration=duration, mood=mood)
        manifest["music"] = music_result
        return manifest

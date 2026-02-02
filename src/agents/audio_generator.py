"""
AudioGenerator Agent pour AIPROD V33
Synthèse vocale (TTS) via ElevenLabs/Google TTS et génération musicale (API externe).
"""
import os
from typing import Any, Dict, Optional


import os
import tempfile
from google.cloud import texttospeech
import requests

class AudioGenerator:
    """
    Génère la voix off (TTS) et la musique de fond pour les vidéos.
    Utilise Google TTS (par défaut) ou ElevenLabs (pour voix avancées/émotions).
    """
    def __init__(self, tts_provider: str = "auto", music_provider: Optional[str] = None):
        self.tts_provider = tts_provider
        self.music_provider = music_provider
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        # GOOGLE_APPLICATION_CREDENTIALS doit être défini dans l'env

    def generate_tts_google(self, text, lang, voice=None, emotion=None):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice or ("fr-FR-Wavenet-D" if lang.startswith("fr") else "en-US-Wavenet-D")
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        fd, output_path = tempfile.mkstemp(suffix=f"_google_{lang}.mp3", prefix="tts_")
        with os.fdopen(fd, "wb") as out:
            out.write(response.audio_content)
        return {"audio_url": output_path, "provider": "google", "lang": lang, "voice": voice, "emotion": emotion}

    def generate_tts_elevenlabs(self, text, lang, voice=None, emotion=None):
        api_key = self.elevenlabs_api_key
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY non défini")
        url = "https://api.elevenlabs.io/v1/text-to-speech"
        payload = {
            "text": text,
            "voice": voice or "Rachel",
            "model_id": "eleven_multilingual_v2"
        }
        headers = {"xi-api-key": api_key}
        response = requests.post(url, json=payload, headers=headers)
        fd, output_path = tempfile.mkstemp(suffix=f"_elevenlabs_{lang}.mp3", prefix="tts_")
        with os.fdopen(fd, "wb") as out:
            out.write(response.content)
        return {"audio_url": output_path, "provider": "elevenlabs", "lang": lang, "voice": voice, "emotion": emotion}

    def generate_tts(self, text: str, lang: str = "fr", voice: Optional[str] = None, emotion: Optional[str] = None) -> Dict[str, Any]:
        """
        Génère un fichier audio à partir d'un texte (TTS) en choisissant le meilleur provider.
        """
        # Sélection dynamique : ElevenLabs si émotion/voix avancée demandée, sinon Google
        try:
            if self.tts_provider == "elevenlabs" or (emotion or voice):
                return self.generate_tts_elevenlabs(text, lang, voice, emotion)
            # Par défaut Google (plus robuste, moins cher)
            return self.generate_tts_google(text, lang, voice, emotion)
        except Exception as e:
            # Fallback : Google si ElevenLabs échoue, ou mock
            try:
                return self.generate_tts_google(text, lang, voice, emotion)
            except Exception:
                return {"audio_url": f"mock_tts_{lang}.mp3", "provider": "mock", "text": text, "lang": lang, "voice": voice, "emotion": emotion, "error": str(e)}

    def generate_music(self, style: str = "cinematic", duration: int = 30) -> Dict[str, Any]:
        """
        Génère une musique de fond adaptée (mock, à remplacer par API réelle).
        """
        # TODO: Intégrer API musicale (AIVA, Soundful, MusicLM...)
        return {"music_url": f"mock_music_{style}_{duration}s.mp3", "provider": self.music_provider or "mock"}

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère les assets audio (TTS + musique) et les ajoute au manifest.
        """
        script = manifest.get("script", "")
        lang = manifest.get("lang", "fr")
        style = manifest.get("music_style", "cinematic")
        duration = manifest.get("duration", 30)
        voice = manifest.get("voice", None)
        emotion = manifest.get("emotion", None)

        tts_result = self.generate_tts(script, lang=lang, voice=voice, emotion=emotion)
        music_result = self.generate_music(style=style, duration=duration)

        manifest["audio"] = tts_result
        manifest["music"] = music_result
        return manifest

"""
VoiceDirector Agent pour AIPROD V33
Gère la direction vocale : émotions, intonation, synchronisation audio/vidéo.
"""
from typing import Any, Dict, Optional

class VoiceDirector:
    """
    Contrôle l’intonation, l’émotion et la synchronisation de la voix off.
    """
    def __init__(self):
        pass

    def map_emotion_to_tts(self, emotion: Optional[str]) -> Dict[str, Any]:
        """
        Mappe une émotion à des paramètres TTS compatibles Google et ElevenLabs.
        """
        # Paramètres Google TTS
        google_mapping = {
            "joy": {"pitch": 2.0, "speaking_rate": 1.1, "volume_gain_db": 2.0},
            "sad": {"pitch": -2.0, "speaking_rate": 0.9, "volume_gain_db": -2.0},
            "angry": {"pitch": 1.0, "speaking_rate": 1.2, "volume_gain_db": 1.0},
            "neutral": {"pitch": 0.0, "speaking_rate": 1.0, "volume_gain_db": 0.0},
        }
        # Paramètres ElevenLabs
        elevenlabs_mapping = {
            "joy": {"stability": 0.3, "similarity_boost": 0.8},
            "sad": {"stability": 0.7, "similarity_boost": 0.5},
            "angry": {"stability": 0.2, "similarity_boost": 0.7},
            "neutral": {"stability": 0.5, "similarity_boost": 0.6},
        }
        e = (emotion or "neutral").lower()
        return {
            "google": google_mapping.get(e, google_mapping["neutral"]),
            "elevenlabs": elevenlabs_mapping.get(e, elevenlabs_mapping["neutral"]),
            "emotion": e
        }

    def synchronize_voice(self, script: str, scenes: list, scene_durations: Optional[list] = None) -> Dict[str, Any]:
        """
        Génère des cues pour synchroniser la voix avec les scènes.
        Si scene_durations est fourni (en secondes), répartit le texte proportionnellement.
        """
        if not scenes:
            return {"cues": []}
        words = script.split()
        n = len(words)
        cues = []
        if scene_durations and len(scene_durations) == len(scenes):
            total_duration = sum(scene_durations)
            idx = 0
            for i, (scene, dur) in enumerate(zip(scenes, scene_durations)):
                word_count = int(round((dur / total_duration) * n)) if total_duration > 0 else n // len(scenes)
                start = idx
                end = min(idx + word_count, n) if i < len(scenes) - 1 else n
                cues.append({
                    "scene": scene,
                    "start_word": start,
                    "end_word": end,
                    "text": " ".join(words[start:end]),
                    "duration": dur
                })
                idx = end
        else:
            per_scene = n // len(scenes)
            idx = 0
            for i, scene in enumerate(scenes):
                start = idx
                end = idx + per_scene if i < len(scenes) - 1 else n
                cues.append({
                    "scene": scene,
                    "start_word": start,
                    "end_word": end,
                    "text": " ".join(words[start:end])
                })
                idx = end
        return {"cues": cues}

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique la direction vocale (émotion, synchronisation, retakes ICC) au manifest.
        """
        script = manifest.get("script", "")
        emotion = manifest.get("emotion", "neutral")
        scenes = manifest.get("scenes", [])
        scene_durations = manifest.get("scene_durations", None)
        retake = manifest.get("voice_retake", None)  # ICC: ajustement demandé ?
        tts_params = self.map_emotion_to_tts(emotion)
        sync = self.synchronize_voice(script, scenes, scene_durations)
        manifest["voice_direction"] = {
            "tts_params": tts_params,
            **sync,
            "retake": retake
        }
        return manifest

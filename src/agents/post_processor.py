"""
PostProcessor Agent pour AIPROD
Montage vidéo, transitions, effets, titrage, sous-titres, mélange audio.
Utilise FFmpeg, OpenCV, PyAV pour les différents effets post-production.
"""

import os
import logging
from src.utils.monitoring import logger
from typing import Any, Dict, Optional, List

try:
    import ffmpeg
except ImportError:
    ffmpeg = None
    logger.warning("[PostProcessor] 'ffmpeg-python' not found. FFmpeg features disabled.")

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("[PostProcessor] 'opencv-python' not found. OpenCV features disabled.")

try:
    import av
except ImportError:
    av = None
    logger.warning("[PostProcessor] 'av' not found. PyAV features disabled.")

try:
    import scenepic
except ImportError:
    scenepic = None
    logger.warning("[PostProcessor] 'scenepic' not found. 3D features disabled.")


class PostProcessor:
    """
    Orchestre le post-traitement complet : transitions, effets vidéo, mélange audio.
    """
    
    def __init__(self, backend: str = "ffmpeg"):
        self.backend = backend
        logger.info(f"[PostProcessor] Initialized with backend: {backend}")

    def apply_transitions(self, video_path: str, transitions: Optional[list] = None) -> str:
        """Applique des transitions vidéo (fade in/out)."""
        if ffmpeg is None:
            logger.warning("[PostProcessor] FFmpeg not available. Skipping transitions.")
            return video_path
        if not transitions:
            return video_path
        
        output = f"trans_{os.path.basename(video_path)}"
        filter_cmds = []
        for t in transitions:
            if t.get('type') == 'fade':
                start = t.get('start', 0)
                duration = t.get('duration', 1)
                filter_cmds.append(f"fade=t=in:st={start}:d={duration}")
        
        if filter_cmds:
            filter_complex = ','.join(filter_cmds)
            try:
                out = ffmpeg.output(
                    ffmpeg.input(video_path),
                    output,
                    vf=filter_complex,
                    vcodec='libx264',
                    acodec='aac'
                )
                ffmpeg.run(out, overwrite_output=True, quiet=True)
                return output
            except Exception as e:
                logger.error(f"[PostProcessor] Error applying transitions: {e}")
                return video_path
        return video_path

    def add_titles_subtitles(self, video_path: str, titles: Optional[list] = None, subtitles: Optional[list] = None) -> str:
        """Ajoute des titres et sous-titres à la vidéo."""
        if ffmpeg is None:
            logger.warning("[PostProcessor] FFmpeg not available. Skipping titles/subtitles.")
            return video_path
        
        output = f"titled_{os.path.basename(video_path)}"
        input_stream = ffmpeg.input(video_path)
        filters = []
        
        if titles:
            for t in titles:
                text = t.get("text", "Title")
                start = t.get("start", 0)
                end = start + t.get("duration", 3)
                filters.append(
                    f"drawtext=text='{text}':fontcolor=white:fontsize=70:x=(w-text_w)/2:y=(h-text_h)/4:enable='between(t,{start},{end})'"
                )
        
        if subtitles:
            for s in subtitles:
                text = s.get("text", "Subtitle")
                start = s.get("start", 0)
                end = start + s.get("duration", 3)
                filters.append(
                    f"drawtext=text='{text}':fontcolor=yellow:fontsize=40:x=(w-text_w)/2:y=h-text_h-40:enable='between(t,{start},{end})'"
                )
        
        if filters:
            filter_complex = ','.join(filters)
            try:
                out = ffmpeg.output(input_stream, output, vf=filter_complex, vcodec='libx264', acodec='aac')
                ffmpeg.run(out, overwrite_output=True, quiet=True)
                return output
            except Exception as e:
                logger.error(f"[PostProcessor] Error adding titles/subtitles: {e}")
                return video_path
        return video_path

    def apply_effects(self, video_path: str, effects: Optional[list] = None) -> str:
        """Applique des effets visuels avec OpenCV."""
        if cv2 is None:
            logger.warning("[PostProcessor] OpenCV not available. Skipping effects.")
            return video_path
        if not effects:
            return video_path
        
        output = f"effects_{os.path.basename(video_path)}"
        try:
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output, fourcc, fps, (width, height))
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                time_sec = frame_idx / fps if fps else 0
                for eff in effects:
                    if eff.get('type') == 'blur' and eff.get('start', 0) <= time_sec <= eff.get('end', 999):
                        frame = cv2.GaussianBlur(frame, (15, 15), 0)
                    if eff.get('type') == 'gray' and eff.get('start', 0) <= time_sec <= eff.get('end', 999):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            return output
        except Exception as e:
            logger.error(f"[PostProcessor] Error applying effects: {e}")
            return video_path

    def apply_pyav_effects(self, video_path: str, effects: Optional[list] = None) -> str:
        """Applique des effets bas niveau avec PyAV."""
        if av is None:
            logger.warning("[PostProcessor] PyAV not available. Skipping PyAV effects.")
            return video_path
        if not effects:
            return video_path
        
        output = f"pyav_{os.path.basename(video_path)}"
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            out = av.open(output, 'w')
            out_stream = out.add_stream('mpeg4', rate=stream.rate)
            out_stream.width = stream.width
            out_stream.height = stream.height
            out_stream.pix_fmt = 'yuv420p'
            
            for frame in container.decode(stream):
                img = frame.to_ndarray(format='bgr24')
                for eff in effects:
                    if eff.get('type') == 'invert':
                        img = 255 - img
                new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                for packet in out_stream.encode(new_frame):
                    out.mux(packet)
            
            out.close()
            container.close()
            return output
        except Exception as e:
            logger.error(f"[PostProcessor] Error applying PyAV effects: {e}")
            return video_path

    def apply_scenepic_overlay(self, video_path: str, overlays: Optional[list] = None) -> str:
        """Ajoute des overlays 3D avec Scenepic."""
        if scenepic is None or not overlays:
            logger.warning("[PostProcessor] Scenepic not available or no overlays.")
            return video_path
        
        output_html = f"scenepic_{os.path.splitext(os.path.basename(video_path))[0]}.html"
        try:
            sp = scenepic.Scene()
            canvas = sp.create_canvas_3d(width=800, height=600)
            mesh = sp.create_mesh(layer_id="cube_layer")
            
            import numpy as np
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            triangles = np.array([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]
            ])
            mesh.add_mesh_without_normals(vertices, triangles)
            sp.link_canvas_events(canvas)
            sp.save_as_html(output_html)
            return output_html
        except Exception as e:
            logger.error(f"[PostProcessor] Error applying scenepic overlay: {e}")
            return video_path

    def mix_audio_tracks(
        self,
        video_path: str,
        audio_tracks: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Mélange plusieurs pistes audio (voix, musique, SFX) en une seule piste.
        
        Args:
            video_path: Chemin du fichier vidéo d'entrée
            audio_tracks: Liste des pistes audio à mixer. Exemple:
                [
                    {"type": "voice", "path": "voice.mp3", "volume": 1.0},
                    {"type": "music", "path": "music.mp3", "volume": 0.7},
                    {"type": "sfx", "path": "sfx.mp3", "volume": 0.5}
                ]
        
        Returns:
            str: Chemin du fichier vidéo avec audio mixé
        """
        if ffmpeg is None:
            logger.warning("[PostProcessor] FFmpeg not available. Skipping audio mixing.")
            return video_path
        
        if not audio_tracks:
            logger.info("[PostProcessor] No audio tracks to mix.")
            return video_path
        
        try:
            # Filtrer les pistes audio existantes
            existing_tracks = [
                t for t in audio_tracks 
                if t.get("path") and os.path.exists(t.get("path", ""))
            ]
            
            if not existing_tracks:
                logger.warning("[PostProcessor] No existing audio files found.")
                return video_path
            
            output = f"mixed_audio_{os.path.basename(video_path)}"
            
            # Construire commande FFmpeg pour mixer l'audio
            input_specs = []
            filter_parts = []
            
            # Input vidéo
            input_specs.append(ffmpeg.input(video_path))
            
            # Inputs audio
            for i, track in enumerate(existing_tracks):
                audio_path = track.get("path")
                volume = track.get("volume", 1.0)
                track_type = track.get("type", "audio")
                
                logger.info(f"[PostProcessor] Adding {track_type} track: {audio_path} (volume: {volume})")
                
                input_audio = ffmpeg.input(audio_path)
                if volume != 1.0:
                    input_audio = ffmpeg.filter(input_audio, 'volume', volume)
                
                input_specs.append(input_audio)
                filter_parts.append(f"[{i+1}]")
            
            # Mixer tous les inputs audio
            if len(existing_tracks) == 1:
                # Un seul track
                output_stream = ffmpeg.output(
                    input_specs[0],  # Vidéo
                    input_specs[1],  # Audio unique
                    output,
                    vcodec='copy',
                    acodec='aac'
                )
            else:
                # Plusieurs tracks - les mixer
                # Construire le filtre amix
                audio_filter = ''.join(filter_parts) + f'amix=inputs={len(existing_tracks)}:duration=longest[a]'
                
                output_stream = ffmpeg.output(
                    input_specs[0]['v'],  # Vidéo
                    ffmpeg.filter(*[inp['a'] for inp in input_specs[1:]], 'amix', inputs=len(existing_tracks), duration='longest'),
                    output,
                    vcodec='copy',
                    acodec='aac'
                )
            
            logger.info(f"[PostProcessor] Running FFmpeg audio mixing ({len(existing_tracks)} tracks)...")
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"[PostProcessor] Audio mixing completed: {output}")
            return output
                
        except Exception as e:
            logger.error(f"[PostProcessor] Error during audio mixing: {e}")
            return video_path

    def composite_audio_with_video(
        self,
        video_path: str,
        audio_path: Optional[str] = None
    ) -> str:
        """
        Attache une piste audio (pré-mixée) à la vidéo.
        """
        if ffmpeg is None:
            logger.warning("[PostProcessor] FFmpeg not available. Skipping audio compositing.")
            return video_path
        
        if not audio_path or not os.path.exists(audio_path):
            logger.warning("[PostProcessor] Audio file not found. Skipping audio compositing.")
            return video_path
        
        try:
            output = f"composite_audio_{os.path.basename(video_path)}"
            
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_path)
            
            output_stream = ffmpeg.output(
                video_input,
                audio_input,
                output,
                vcodec='copy',
                acodec='aac'
            )
            
            logger.info(f"[PostProcessor] Compositing audio with video...")
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"[PostProcessor] Audio compositing completed: {output}")
            return output
            
        except Exception as e:
            logger.error(f"[PostProcessor] Error during audio compositing: {e}")
            return video_path

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique le post-traitement complet : transitions, effets vidéo, mélange audio.

        Args:
            manifest: Dictionnaire avec vidéo et paramètres audio

        Returns:
            Dict enrichi avec chemin fichier post-traité
        """
        video_path = manifest.get("video_path", "output_video.mp4")
        transitions = manifest.get("transitions", [])
        titles = manifest.get("titles", [])
        subtitles = manifest.get("subtitles", [])
        effects = manifest.get("effects", [])
        overlays = manifest.get("overlays", [])
        audio_tracks = manifest.get("audio_tracks", [])
        
        logger.info(f"[PostProcessor] Starting post-processing: {video_path}")

        # Étape 1: Transitions vidéo
        if transitions:
            video_path = self.apply_transitions(video_path, transitions)
            logger.info("[PostProcessor] ✓ Transitions applied")
        
        # Étape 2: Titres et sous-titres
        if titles or subtitles:
            video_path = self.add_titles_subtitles(video_path, titles, subtitles)
            logger.info("[PostProcessor] ✓ Titles/subtitles added")
        
        # Étape 3: Effets vidéo (OpenCV)
        if effects:
            video_path = self.apply_effects(video_path, effects)
            logger.info("[PostProcessor] ✓ Video effects applied")
        
        # Étape 4: Effets PyAV
        if effects:
            video_path = self.apply_pyav_effects(video_path, effects)
            logger.info("[PostProcessor] ✓ PyAV effects applied")
        
        # Étape 5: Overlays 3D
        if overlays:
            video_path = self.apply_scenepic_overlay(video_path, overlays)
            logger.info("[PostProcessor] ✓ 3D overlays added")
        
        # Étape 6: Mélange audio (voix + musique + SFX)
        if audio_tracks:
            logger.info(f"[PostProcessor] Mixing {len(audio_tracks)} audio tracks...")
            video_path = self.mix_audio_tracks(video_path, audio_tracks)
            logger.info(f"[PostProcessor] ✓ Audio mixing completed ({len(audio_tracks)} tracks)")

        manifest["post_processed_video"] = video_path
        logger.info(f"[PostProcessor] ✓ Post-processing complete!")
        return manifest

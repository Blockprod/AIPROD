"""
PostProcessor Agent pour AIPROD V33
Montage vidéo, transitions, effets, titrage, sous-titres (MoviePy/FFmpeg ou API cloud).
"""

import os
from src.utils.monitoring import logger
from typing import Any, Dict, Optional
try:
    import ffmpeg
except ImportError:
    ffmpeg = None
    logger.warning("[PostProcessor] 'ffmpeg-python' package not found. FFmpeg features will be disabled. To enable, install with: pip install ffmpeg-python")
try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("[PostProcessor] 'opencv-python' package not found. OpenCV features will be disabled. To enable, install with: pip install opencv-python")
try:
    import av
except ImportError:
    av = None
    logger.warning("[PostProcessor] 'av' package not found. PyAV features will be disabled. To enable, install with: pip install av")
try:
    import scenepic
except ImportError:
    scenepic = None
    logger.warning("[PostProcessor] 'scenepic' package not found. 3D overlay features will be disabled. To enable, install with: pip install scenepic")

class PostProcessor:
    """
    Applique le montage, transitions, effets visuels, titrage et sous-titres à la vidéo générée.
    """
    def __init__(self, backend: str = "ffmpeg"):
        self.backend = backend  # "ffmpeg", "opencv", "pyav", "scenepic", ou "cloud"

    def apply_transitions(self, video_path: str, transitions: Optional[list] = None) -> str:
        """
        Applique des transitions entre scènes à l'aide de ffmpeg-python.

        Args:
            video_path (str): Chemin du fichier vidéo d'entrée.
            transitions (Optional[list]): Liste de transitions à appliquer. Exemple :
                [{"type": "fade", "start": 5, "duration": 2}, ...]

        Returns:
            str: Chemin du fichier vidéo de sortie (avec transitions appliquées), ou le chemin d'origine en cas d'échec.
        """
        if ffmpeg is None:
            logger.warning("[PostProcessor] FFmpeg not available. Skipping transitions.")
            return video_path
        if not transitions:
            return video_path
        output = f"trans_{os.path.basename(video_path)}"
        # Example: fade in/out using ffmpeg filter_complex
        filter_cmds = []
        for t in transitions:
            if t.get('type') == 'fade':
                start = t.get('start', 0)
                duration = t.get('duration', 1)
                filter_cmds.append(f"fade=t=in:st={start}:d={duration}")
        if filter_cmds:
            filter_complex = ','.join(filter_cmds)
            out = ffmpeg.output(ffmpeg.input(video_path), output, vf=filter_complex, vcodec='libx264', acodec='aac')
            try:
                ffmpeg.run(out, overwrite_output=True, quiet=True)
                return output
            except ffmpeg.Error as e:
                logger.error(f"[PostProcessor] Erreur FFmpeg transitions: {e}")
                return video_path
        return video_path

    def add_titles_subtitles(self, video_path: str, titles: Optional[list] = None, subtitles: Optional[list] = None) -> str:
        """
        Ajoute des titres et/ou sous-titres à la vidéo à l'aide de ffmpeg-python (overlay texte).

        Args:
            video_path (str): Chemin du fichier vidéo d'entrée.
            titles (Optional[list]): Liste de titres à ajouter. Exemple :
                [{"text": "Titre", "start": 0, "duration": 3}, ...]
            subtitles (Optional[list]): Liste de sous-titres à ajouter. Exemple :
                [{"text": "Sous-titre", "start": 0, "duration": 3}, ...]

        Returns:
            str: Chemin du fichier vidéo de sortie (avec titres/sous-titres), ou le chemin d'origine en cas d'échec.
        """
        if ffmpeg is None:
            logger.warning("[PostProcessor] FFmpeg not available. Skipping titles/subtitles.")
            return video_path
        output = f"postproc_{os.path.basename(video_path)}"
        input_stream = ffmpeg.input(video_path)
        filters = []
        if titles:
            for t in titles:
                text = t.get("text", "Titre")
                start = t.get("start", 0)
                end = start + t.get("duration", 3)
                filters.append(
                    f"drawtext=text='{text}':fontcolor=white:fontsize=70:x=(w-text_w)/2:y=(h-text_h)/4:enable='between(t,{start},{end})'"
                )
        if subtitles:
            for s in subtitles:
                text = s.get("text", "Sous-titre")
                start = s.get("start", 0)
                end = start + s.get("duration", 3)
                filters.append(
                    f"drawtext=text='{text}':fontcolor=yellow:fontsize=40:x=(w-text_w)/2:y=h-text_h-40:enable='between(t,{start},{end})'"
                )
        if filters:
            filter_complex = ','.join(filters)
            out = ffmpeg.output(input_stream, output, vf=filter_complex, vcodec='libx264', acodec='aac')
        else:
            out = ffmpeg.output(input_stream, output, vcodec='libx264', acodec='aac')
        try:
            ffmpeg.run(out, overwrite_output=True, quiet=True)
            return output
        except ffmpeg.Error as e:
            logger.error(f"[PostProcessor] Erreur FFmpeg: {e}")
            return video_path

    def apply_effects(self, video_path: str, effects: Optional[list] = None) -> str:
        """
        Applique des effets visuels à la vidéo avec OpenCV, optimisé par multiprocessing (batch frames).

        Args:
            video_path (str): Chemin du fichier vidéo d'entrée.
            effects (Optional[list]): Liste d'effets à appliquer. Exemple :
                [{"type": "blur", "start": 2, "end": 5}, ...]

        Returns:
            str: Chemin du fichier vidéo de sortie (avec effets appliqués), ou le chemin d'origine en cas d'échec.
        """
        import numpy as np
        from multiprocessing import Pool, cpu_count

        def process_frame(args):
            frame_idx, frame, fps, effects = args
            time_sec = frame_idx / fps if fps else 0
            if cv2 is None:
                return frame_idx, frame
            for eff in effects:
                if eff.get('type') == 'blur' and eff.get('start', 0) <= time_sec <= eff.get('end', 0):
                    frame = cv2.GaussianBlur(frame, (15, 15), 0)
                if eff.get('type') == 'gray' and eff.get('start', 0) <= time_sec <= eff.get('end', 0):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame_idx, frame

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
            frames = []
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append((frame_idx, frame, fps, effects))
                frame_idx += 1
            cap.release()
            # Traitement en parallèle
            try:
                with Pool(processes=cpu_count()) as pool:
                    results = pool.map(process_frame, frames)
                results.sort()  # Assure l'ordre des frames
                processed_frames = [f for idx, f in results]
            except Exception as e:
                logger.warning(f"[PostProcessor] Multiprocessing failed, fallback to sequential: {e}")
                processed_frames = [process_frame(args)[1] for args in frames]
            out = cv2.VideoWriter(output, fourcc, fps, (width, height))
            for frame in processed_frames:
                out.write(frame)
            out.release()
            return output
        except Exception as e:
            logger.error(f"[PostProcessor] Erreur OpenCV: {e}")
            return video_path
    def apply_pyav_effects(self, video_path: str, effects: Optional[list] = None) -> str:
        """
        Applique des effets bas niveau à la vidéo avec PyAV (exemple : inversion des couleurs).

        Args:
            video_path (str): Chemin du fichier vidéo d'entrée.
            effects (Optional[list]): Liste d'effets à appliquer. Exemple :
                [{"type": "invert"}]

        Returns:
            str: Chemin du fichier vidéo de sortie (avec effets PyAV appliqués), ou le chemin d'origine en cas d'échec.
        """
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
            logger.error(f"[PostProcessor] Erreur PyAV: {e}")
            return video_path
    def apply_scenepic_overlay(self, video_path: str, overlays: Optional[list] = None) -> str:
        """
        Ajoute une animation/scène 3D avec Scenepic (si installé).

        Args:
            video_path (str): Chemin du fichier vidéo d'entrée.
            overlays (Optional[list]): Liste d'éléments 3D à ajouter (non utilisé dans cet exemple).

        Returns:
            str: Chemin du fichier HTML généré (visualisation 3D), ou le chemin d'origine en cas d'échec.
        """
        if scenepic is None or not overlays:
            logger.warning("[PostProcessor] Scenepic not available or no overlays. Skipping 3D overlay.")
            return video_path
        output_html = f"scenepic_{os.path.splitext(os.path.basename(video_path))[0]}.html"
        try:
            sp = scenepic.Scene()
            canvas = sp.create_canvas_3d(width=800, height=600)
            mesh = sp.create_mesh(layer_id="cube_layer")
            # Ajout d'un cube simple
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
            logger.error(f"[PostProcessor] Erreur Scenepic: {e}")
            return video_path

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique le post-traitement à la vidéo finale avec tous les backends disponibles.

        Args:
            manifest (Dict[str, Any]): Dictionnaire contenant les chemins et paramètres de la vidéo à traiter.

        Returns:
            Dict[str, Any]: Le manifest enrichi avec le chemin du fichier post-traité (clé 'post_processed_video').
        """
        video_path = manifest.get("video_path", "output_video.mp4")
        transitions = manifest.get("transitions", [])
        titles = manifest.get("titles", [])
        subtitles = manifest.get("subtitles", [])
        effects = manifest.get("effects", [])
        overlays = manifest.get("overlays", [])

        # FFmpeg pour transitions et overlays texte
        video_path = self.apply_transitions(video_path, transitions)
        video_path = self.add_titles_subtitles(video_path, titles, subtitles)
        # OpenCV pour effets frame par frame
        video_path = self.apply_effects(video_path, effects)
        # PyAV pour effets bas niveau
        video_path = self.apply_pyav_effects(video_path, effects)
        # Scenepic pour overlays 3D (retourne un HTML si overlay, sinon la vidéo)
        video_path = self.apply_scenepic_overlay(video_path, overlays)

        manifest["post_processed_video"] = video_path
        return manifest
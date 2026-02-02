import os
import pytest
from src.agents.post_processor import PostProcessor
import numpy as np

@pytest.mark.parametrize("backend", ["ffmpeg", "opencv", "pyav", "scenepic"])
def test_postprocessor_multibackend(tmp_path, backend):
    """
    Test d'intégration : vérifie que le PostProcessor fonctionne avec chaque backend,
    produit un fichier de sortie et ne plante pas même si une dépendance est manquante.
    """
    # Prépare une vidéo d'entrée factice (ou utilise un petit fichier de test existant)
    input_video = str(tmp_path / "input.mp4")
    # Génère une vidéo noire de 1s (si OpenCV dispo)
    try:
        import cv2
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(input_video, fourcc, 24, (64, 64))
        for _ in range(24):
            out.write((0 * np.ones((64, 64, 3), dtype='uint8')))
        out.release()
    except Exception:
        # Si OpenCV non dispo, skip le test
        pytest.skip("OpenCV requis pour générer la vidéo de test")

    # Effets simples pour chaque backend
    effects = [{"type": "blur", "start": 0, "end": 1}] if backend == "opencv" else []
    overlays = [{}] if backend == "scenepic" else []
    transitions = [{"type": "fade", "start": 0, "duration": 1}] if backend == "ffmpeg" else []

    pp = PostProcessor(backend=backend)
    # Appelle chaque méthode selon le backend
    if backend == "ffmpeg":
        out_path = pp.apply_transitions(input_video, transitions)
    elif backend == "opencv":
        out_path = pp.apply_effects(input_video, effects)
    elif backend == "pyav":
        out_path = pp.apply_pyav_effects(input_video, effects)
    elif backend == "scenepic":
        out_path = pp.apply_scenepic_overlay(input_video, overlays)
    else:
        pytest.skip(f"Backend inconnu : {backend}")

    assert os.path.exists(out_path), f"Le fichier de sortie n'a pas été généré pour {backend}"

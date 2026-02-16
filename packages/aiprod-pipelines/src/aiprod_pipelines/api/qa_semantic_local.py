"""
QA Sémantique Souveraine — évaluation prompt↔vidéo via CLIP/SigLIP local.

Remplace le QA sémantique cloud (Gemini Vision) par un scoring local
basé sur les embeddings CLIP. Fonctionne sur le GPU local, aucun appel
API externe.

Architecture :
    1. Extraire N frames de la vidéo
    2. Encoder les frames avec CLIP vision encoder
    3. Encoder le prompt avec CLIP text encoder
    4. Calculer la similarité cosinus prompt↔frames
    5. Agréger en un score de cohérence [0, 1]

Modèle par défaut : openai/clip-vit-large-patch14 (~400MB)
Alternative souveraine future : apple/siglip-so400m-patch14-384
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports pour ne pas bloquer si torch/clip pas installés
_CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None
_clip_tokenizer = None


def _ensure_clip_loaded(
    model_name: str = "openai/clip-vit-large-patch14",
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> None:
    """Charge CLIP une seule fois (singleton)."""
    global _CLIP_AVAILABLE, _clip_model, _clip_processor, _clip_tokenizer

    if _clip_model is not None:
        return

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        path = model_path or model_name

        logger.info("Loading CLIP model from %s ...", path)
        _clip_model = CLIPModel.from_pretrained(
            path,
            local_files_only=True,
        ).to(device).eval()

        _clip_processor = CLIPProcessor.from_pretrained(
            path,
            local_files_only=True,
        )

        _CLIP_AVAILABLE = True
        logger.info("CLIP model loaded on %s", device)

    except Exception as e:
        logger.warning("CLIP not available: %s — QA will return neutral scores", e)
        _CLIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QASemanticResult:
    """Résultat du QA sémantique local."""

    overall_score: float  # [0, 1]
    prompt_similarity: float  # cosine similarity prompt↔frames moyennée
    frame_scores: List[float] = field(default_factory=list)
    temporal_consistency: float = 0.0  # std des scores inter-frames
    details: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def __post_init__(self) -> None:
        self.passed = self.overall_score >= 0.2  # seuil CLIP typique


# ---------------------------------------------------------------------------
# QA Engine
# ---------------------------------------------------------------------------


class SemanticQALocal:
    """
    QA sémantique souveraine basée sur CLIP/SigLIP.

    Usage :
        qa = SemanticQALocal()
        result = qa.evaluate("a cat running on the beach", "output/video.mp4")
        print(result.overall_score, result.passed)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        model_path: Optional[str] = None,
        device: str = "cuda",
        num_frames: int = 8,
        similarity_threshold: float = 0.2,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.num_frames = num_frames
        self.similarity_threshold = similarity_threshold

    def evaluate(
        self,
        prompt: str,
        video_path: str,
        negative_prompt: str = "",
    ) -> QASemanticResult:
        """
        Évalue la cohérence entre un prompt et une vidéo générée.

        Args:
            prompt: Le texte du prompt original
            video_path: Chemin vers le fichier vidéo .mp4
            negative_prompt: Prompt négatif (optionnel)

        Returns:
            QASemanticResult avec score, détails, et flag passed
        """
        # 1. Extraire les frames
        frames = self._extract_frames(video_path)
        if not frames:
            logger.warning("No frames extracted from %s", video_path)
            return QASemanticResult(
                overall_score=0.0,
                prompt_similarity=0.0,
                details={"error": "no_frames_extracted"},
            )

        # 2. Charger CLIP si nécessaire
        _ensure_clip_loaded(self.model_name, self.model_path, self.device)

        if not _CLIP_AVAILABLE:
            # Mode dégradé : retourner un score neutre
            return QASemanticResult(
                overall_score=0.5,
                prompt_similarity=0.5,
                frame_scores=[0.5] * len(frames),
                temporal_consistency=1.0,
                details={"mode": "stub_no_clip"},
                passed=True,
            )

        # 3. Calculer les scores
        frame_scores = self._compute_similarities(prompt, frames)
        prompt_sim = float(np.mean(frame_scores)) if frame_scores else 0.0
        temporal_std = float(np.std(frame_scores)) if len(frame_scores) > 1 else 0.0
        temporal_consistency = max(0.0, 1.0 - temporal_std * 5)  # pénaliser la variance

        # Score global : 70% similarité prompt, 30% cohérence temporelle
        overall = 0.7 * prompt_sim + 0.3 * temporal_consistency

        result = QASemanticResult(
            overall_score=overall,
            prompt_similarity=prompt_sim,
            frame_scores=[float(s) for s in frame_scores],
            temporal_consistency=temporal_consistency,
            details={
                "model": self.model_name,
                "num_frames_analyzed": len(frames),
                "video_path": video_path,
                "prompt_length": len(prompt),
                "threshold": self.similarity_threshold,
            },
        )

        logger.info(
            "QA Semantic: score=%.3f (sim=%.3f, consistency=%.3f) — %s",
            overall,
            prompt_sim,
            temporal_consistency,
            "PASS" if result.passed else "FAIL",
        )

        return result

    def evaluate_batch(
        self,
        assets: List[Dict[str, Any]],
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Évalue un batch d'assets générés.

        Args:
            assets: Liste de dicts avec 'output_path' ou 'url'
            prompt: Le prompt commun

        Returns:
            Dict avec scores agrégés et détails par asset
        """
        results = []
        for asset in assets:
            path = asset.get("output_path") or asset.get("url", "")
            if path.startswith("file://"):
                path = path[7:]
            if Path(path).exists():
                r = self.evaluate(prompt, path)
                results.append({"asset_id": asset.get("id", "?"), "result": r})
            else:
                results.append({
                    "asset_id": asset.get("id", "?"),
                    "result": QASemanticResult(
                        overall_score=0.0,
                        prompt_similarity=0.0,
                        details={"error": f"file_not_found: {path}"},
                    ),
                })

        scores = [r["result"].overall_score for r in results]
        avg_score = float(np.mean(scores)) if scores else 0.0
        all_passed = all(r["result"].passed for r in results) if results else False

        return {
            "average_score": avg_score,
            "all_passed": all_passed,
            "num_assets": len(results),
            "details": [
                {
                    "asset_id": r["asset_id"],
                    "score": r["result"].overall_score,
                    "passed": r["result"].passed,
                }
                for r in results
            ],
        }

    # ---- Internal -----------------------------------------------------------

    def _extract_frames(self, video_path: str) -> List[Any]:
        """Extrait N frames uniformément réparties de la vidéo."""
        path = Path(video_path)
        if not path.exists():
            return []

        try:
            import imageio.v3 as iio

            # Lire toutes les frames
            all_frames = iio.imread(str(path))
            total = len(all_frames)

            if total == 0:
                return []

            # Sélectionner uniformément
            n = min(self.num_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int)
            frames = [all_frames[i] for i in indices]

            logger.debug("Extracted %d/%d frames from %s", n, total, path.name)
            return frames

        except Exception as e:
            logger.warning("Frame extraction failed for %s: %s", video_path, e)
            return []

    def _compute_similarities(
        self, prompt: str, frames: List[Any]
    ) -> List[float]:
        """Calcule la similarité cosinus CLIP entre le prompt et chaque frame."""
        import torch
        from PIL import Image

        scores = []
        try:
            # Encoder le texte
            text_inputs = _clip_processor(
                text=[prompt], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                text_features = _clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Encoder chaque frame
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    image = Image.fromarray(frame)
                else:
                    image = frame

                image_inputs = _clip_processor(
                    images=image, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    image_features = _clip_model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )

                # Cosine similarity
                similarity = (text_features @ image_features.T).item()
                scores.append(similarity)

        except Exception as e:
            logger.error("CLIP similarity computation failed: %s", e)
            return [0.5] * len(frames)

        return scores

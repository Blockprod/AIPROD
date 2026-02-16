"""
Reproducibility module — Garantit la reproductibilité bit-exact des inférences.

Usage dans le pipeline :
    from aiprod_pipelines.utils.reproducibility import set_deterministic_mode

    set_deterministic_mode(seed=42)
    # ... pipeline inference ...
"""

from __future__ import annotations

import os
import logging
import random

logger = logging.getLogger(__name__)


def set_deterministic_mode(seed: int = 42) -> None:
    """
    Configure l'environnement pour une exécution déterministe.

    Active :
    - torch.manual_seed + torch.cuda.manual_seed_all
    - numpy seed
    - random seed
    - cuDNN benchmark=False, deterministic=True
    - CUBLAS_WORKSPACE_CONFIG pour les algos déterministes
    - torch.use_deterministic_algorithms (warn_only)

    Args:
        seed: Graine de reproductibilité
    """
    import numpy as np
    import torch

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU + GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS workspace config pour ops déterministes
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Global deterministic mode (warn_only pour ops sans implémentation déterministe)
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("Deterministic mode enabled — seed=%d", seed)


def get_reproducibility_info() -> dict:
    """
    Retourne les informations de reproductibilité de l'environnement courant.

    Returns:
        Dict avec versions torch, cuda, cudnn, et flags déterministes.
    """
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hashseed": os.environ.get("PYTHONHASHSEED"),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info

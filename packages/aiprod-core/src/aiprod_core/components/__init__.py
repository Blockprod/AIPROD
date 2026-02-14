# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Components — Scheduler, Guider, Diffusion Steps

Core building blocks for the diffusion pipeline:
    - AdaptiveFlowScheduler: learned noise scheduling
    - ClassifierFreeGuider: CFG with configurable rescaling
    - EulerFlowStep: single denoising step using Euler method
"""

from .scheduler import AdaptiveFlowScheduler
from .guider import ClassifierFreeGuider, CFGGuider, STGGuider, MultiModalGuider, MultiModalGuiderParams
from .diffusion_step import EulerFlowStep
from .noiser import GaussianNoiser
from .patchifier import VideoLatentPatchifier, AudioPatchifier, get_pixel_coords
from .protocols import DiffusionStepProtocol, GuiderProtocol, NoiserProtocol

# Alias for pipeline compatibility
AIPROD2Scheduler = AdaptiveFlowScheduler
Noiser = GaussianNoiser

__all__ = [
    "AdaptiveFlowScheduler",
    "AIPROD2Scheduler",
    "ClassifierFreeGuider",
    "CFGGuider",
    "STGGuider",
    "MultiModalGuider",
    "MultiModalGuiderParams",
    "EulerFlowStep",
    "GaussianNoiser",
    "Noiser",
    "VideoLatentPatchifier",
    "AudioPatchifier",
    "get_pixel_coords",
    "DiffusionStepProtocol",
    "GuiderProtocol",
    "NoiserProtocol",
]

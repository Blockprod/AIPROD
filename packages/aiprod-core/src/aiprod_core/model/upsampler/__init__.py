"""Latent upsampler model components."""

from aiprod_core.model.upsampler.model import LatentUpsampler, upsample_video
from aiprod_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]

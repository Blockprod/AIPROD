# Backward-compat shim — old import path
# Use `aiprod_core.components` for new code.
from .patchifier import VideoLatentPatchifier, AudioPatchifier, get_pixel_coords

__all__ = ["VideoLatentPatchifier", "AudioPatchifier", "get_pixel_coords"]

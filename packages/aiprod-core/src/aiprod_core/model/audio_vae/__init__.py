# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
NAC — Neural Audio Codec

AIPROD's proprietary audio encoding/decoding, using a residual
vector-quantized (RVQ) codec architecture. Distinct from HiFi-GAN
vocoder-based approaches.

Key features:
    - RVQ-based discrete audio tokens (not continuous latents)
    - Multi-band spectral processing
    - Configurable bitrate (1.5-24 kbps)
    - 48kHz support
"""

from .codec import AudioEncoder, AudioDecoder, NACConfig

__all__ = ["AudioEncoder", "AudioDecoder", "NACConfig"]

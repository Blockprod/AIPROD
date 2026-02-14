# Backward-compat shim — old import path
# Use `aiprod_core.components` for new code.
from .noiser import GaussianNoiser

Noiser = GaussianNoiser

__all__ = ["GaussianNoiser", "Noiser"]

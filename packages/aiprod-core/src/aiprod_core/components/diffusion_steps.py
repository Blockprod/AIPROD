# Backward-compat shim — old import path
# Use `aiprod_core.components` for new code.
from .diffusion_step import EulerFlowStep

# Legacy alias
EulerDiffusionStep = EulerFlowStep

__all__ = ["EulerDiffusionStep", "EulerFlowStep"]

# Backward-compat shim — old import path
# Use `aiprod_core.components` for new code.
from .scheduler import AdaptiveFlowScheduler

AIPROD2Scheduler = AdaptiveFlowScheduler

__all__ = ["AdaptiveFlowScheduler", "AIPROD2Scheduler"]

"""AIPROD v2 Training Infrastructure

Curriculum Learning Strategy:
- Phase 1: Simple objects (single moving subject, clean backgrounds)
- Phase 2: Compound scenes (2-3 subjects, moderate complexity)
- Phase 3: Complex motion (fast action, occlusions, perspective changes)
- Phase 4: Edge cases (unusual angles, weather, dynamic lighting)
- Phase 5: Refinement (mix of all, fine-tuning)

Timeline: 6-8 weeks on GTX 1070 with ~100-150 hours curated video
Expected Quality: FVD ~30 (professional grade)
"""

from .curriculum import CurriculumTrainer, TrainingPhase, CurriculumConfig

__all__ = ["CurriculumTrainer", "TrainingPhase", "CurriculumConfig"]

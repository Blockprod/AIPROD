#!/usr/bin/env python
"""
Test runner for PHASE 1 integration tests.
Handles proper path setup to avoid import issues.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
aiprod_core_src = project_root / "packages" / "aiprod-core" / "src"
aiprod_pipelines_src = project_root / "packages" / "aiprod-pipelines" / "src"

sys.path.insert(0, str(aiprod_core_src))
sys.path.insert(0, str(aiprod_pipelines_src))

# Now run pytest
import pytest

if __name__ == "__main__":
    # Run all PHASE 1 tests
    test_file = Path(__file__).parent / "tests" / "test_foundation.py"
    exit_code = pytest.main([str(test_file), "-v", "--tb=short", "-k", "TestInputSanitizer or TestCreativeDirector or TestVisualTranslator or TestRenderExecutor or TestPipeline"])
    sys.exit(exit_code)

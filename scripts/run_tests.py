#!/usr/bin/env python3
"""
Simple test runner for AIPROD
Provides alternative to pytest with coverage support
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", with_coverage=False):
    """Run tests with optional coverage."""

    test_paths = {
        "all": "tests/",
        "unit": "tests/unit/",
        "integration": "tests/integration/",
        "performance": "tests/performance/",
    }

    path = test_paths.get(test_type, "tests/")

    # Build pytest command
    cmd = ["python", "-m", "pytest", path, "-v", "--tb=short"]

    if with_coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])

    print(f"\n{'='*80}")
    print(f"🚀 Running tests: {test_type}")
    print(f"{'='*80}\n")
    print(f"Command: {' '.join(cmd)}\n")

    # Run pytest
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    return result.returncode


def main():
    """Main entry point."""

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        with_coverage = "--cov" in sys.argv
    else:
        test_type = "all"
        with_coverage = True

    exit_code = run_tests(test_type, with_coverage)

    print(f"\n{'='*80}")
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print(f"{'='*80}\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

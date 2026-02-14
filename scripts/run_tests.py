#!/usr/bin/env python3
"""Script to run all tests and generate coverage reports."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run pytest with coverage reporting."""
    
    repo_root = Path(__file__).parent
    
    print("=" * 70)
    print("AIPROD Test Suite Runner")
    print("=" * 70)
    
    # Test modules to cover
    test_modules = [
        "aiprod_core",
        "aiprod_trainer",
    ]
    
    # Run pytest with coverage
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=aiprod_core",
        "--cov=aiprod_trainer",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v",
        "--tb=short",
    ]
    
    print("\nRunning command:")
    print(" ".join(command))
    print("\n")
    
    result = subprocess.run(command, cwd=repo_root)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        print(f"\nCoverage report generated in: htmlcov/index.html")
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed")
        print("=" * 70)
        return result.returncode
    
    return 0


def run_unit_tests_only():
    """Run only unit tests (no GPU/slow tests)."""
    
    repo_root = Path(__file__).parent
    
    print("=" * 70)
    print("AIPROD Unit Tests (no GPU, no slow tests)")
    print("=" * 70)
    
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m", "not gpu and not slow",
        "--cov=aiprod_core",
        "--cov=aiprod_trainer",
        "--cov-report=term-missing",
        "-v",
    ]
    
    print("\nRunning command:")
    print(" ".join(command))
    print("\n")
    
    result = subprocess.run(command, cwd=repo_root)
    return result.returncode


def run_integration_tests():
    """Run integration tests."""
    
    repo_root = Path(__file__).parent
    
    print("=" * 70)
    print("AIPROD Integration Tests")
    print("=" * 70)
    
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m", "integration",
        "-v",
        "-s",
    ]
    
    print("\nRunning command:")
    print(" ".join(command))
    print("\n")
    
    result = subprocess.run(command, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AIPROD tests")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "gpu"],
        default="unit",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    if args.type == "all":
        exit_code = run_tests()
    elif args.type == "unit":
        exit_code = run_unit_tests_only()
    elif args.type == "integration":
        exit_code = run_integration_tests()
    else:
        exit_code = 1
    
    sys.exit(exit_code)

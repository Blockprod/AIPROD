"""
Test runner and reporting for streaming module.

Usage:
    pytest tests/streaming/ -v
    pytest tests/streaming/ --benchmark-only
    pytest tests/streaming/ --cov=aiprod_trainer.streaming
"""

import subprocess
import sys
from pathlib import Path


def run_tests(
    test_type: str = "all",
    verbose: bool = True,
    coverage: bool = False,
) -> int:
    """
    Run streaming tests.
    
    Args:
        test_type: 'all', 'unit', 'integration', 'benchmark'
        verbose: Print verbose output
        coverage: Generate coverage report
        
    Returns:
        Exit code (0 = success)
    """
    test_dir = Path(__file__).parent / "tests" / "streaming"
    
    cmd = ["pytest", str(test_dir)]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=aiprod_trainer.streaming",
            "--cov-report=html",
            "--cov-report=term",
        ])
    
    if test_type == "unit":
        cmd.extend(["-m", "not integration and not benchmark"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "benchmark":
        cmd.append("--benchmark-only")
    
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"
    
    exit_code = run_tests(test_type=test_type, coverage=True)
    sys.exit(exit_code)

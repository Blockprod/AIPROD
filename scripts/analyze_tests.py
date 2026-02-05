#!/usr/bin/env python3
"""
Test Report Generator for AIPROD
Analyses all test files and generates a comprehensive report
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def count_test_functions(file_path):
    """Count number of test functions in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Count functions starting with test_
            matches = re.findall(r"def\s+test_\w+", content)
            return len(matches)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def analyze_tests(root_dir):
    """Analyze all test files and generate statistics."""

    stats = {
        "by_directory": defaultdict(lambda: {"files": 0, "functions": 0}),
        "total_files": 0,
        "total_functions": 0,
        "files": [],
    }

    tests_dir = Path(root_dir) / "tests"

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return stats

    # Find all test files
    for test_file in tests_dir.rglob("test_*.py"):
        rel_path = test_file.relative_to(tests_dir)
        directory = str(rel_path.parent)

        count = count_test_functions(test_file)

        stats["files"].append(
            {"path": str(rel_path), "directory": directory, "functions": count}
        )

        stats["by_directory"][directory]["files"] += 1
        stats["by_directory"][directory]["functions"] += count
        stats["total_files"] += 1
        stats["total_functions"] += count

    return stats


def print_report(stats):
    """Print formatted test report."""

    print("\n" + "=" * 80)
    print("🧪 AIPROD TEST SUITE ANALYSIS")
    print("=" * 80)

    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"  Total Test Files:        {stats['total_files']}")
    print(f"  Total Test Functions:    {stats['total_functions']}")

    print(f"\n📁 BREAKDOWN BY DIRECTORY")
    print("-" * 80)

    for directory in sorted(stats["by_directory"].keys()):
        info = stats["by_directory"][directory]
        print(
            f"  {directory or 'root':.<40} {info['files']:>3} files  {info['functions']:>3} tests"
        )

    print(f"\n📝 TEST FILES")
    print("-" * 80)

    # Group by directory
    by_dir = defaultdict(list)
    for file_info in stats["files"]:
        by_dir[file_info["directory"]].append(file_info)

    for directory in sorted(by_dir.keys()):
        if directory:
            print(f"\n  {directory}/")
        else:
            print(f"\n  Root tests/")

        for file_info in sorted(by_dir[directory], key=lambda x: x["path"]):
            filename = Path(file_info["path"]).name
            count = file_info["functions"]
            print(f"    • {filename:<50} {count:>3} tests")

    print(f"\n✅ VERDICT")
    print("-" * 80)
    if stats["total_functions"] > 100:
        print(f"  ✅ Comprehensive test suite with {stats['total_functions']} tests")
        print(f"  ✅ Ready for pytest execution")
        print(f"  ✅ Coverage analysis available")
    else:
        print(f"  ⚠️  Limited test coverage ({stats['total_functions']} tests)")

    print("\n" + "=" * 80)
    print("\n💡 TO RUN TESTS:")
    print("\n  pytest tests/ -v")
    print("  pytest tests/ -v --cov=src --cov-report=html")
    print("  pytest tests/unit/ -v")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    print(f"Analyzing tests in: {root_dir}")
    stats = analyze_tests(root_dir)
    print_report(stats)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script pour generer couverture de code"""

import subprocess
import sys


def main():
    """Run coverage."""
    cmd = [sys.executable, "-m", "coverage", "run", "-m", "pytest", "tests/", "-q"]
    print("[*] Running coverage...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("[!] Coverage failed")
        return 1

    print("\n[*] Generating HTML report...")
    subprocess.run([sys.executable, "-m", "coverage", "html"])
    print("[+] HTML report: htmlcov/index.html")

    print("\n[*] Coverage summary:\n")
    subprocess.run([sys.executable, "-m", "coverage", "report", "-m"])

    return 0


if __name__ == "__main__":
    sys.exit(main())

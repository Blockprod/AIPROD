#!/usr/bin/env python3
"""
🔐 AIPROD Security Audit Script
Scans the entire codebase for exposed secrets and security issues.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns to detect exposed secrets
SECRET_PATTERNS = {
    "GEMINI_API_KEY": r"AIzaSy[A-Za-z0-9_-]{40,}",
    "RUNWAY_API_KEY": r"key_[a-f0-9]{60,}",
    "REPLICATE_API_KEY": r"r8_[A-Za-z0-9_]{30,}",
    "ELEVENLABS_API_KEY": r"sk_[a-f0-9]+",
    "DATABASE_PASSWORD": r"password\s*[=:]\s*(['\"]?)([^'\";\s]+)\1",
    "HARDCODED_PASSWORD": r"password@",
}

# Dangerous patterns
DANGEROUS_PATTERNS = {
    "Hardcoded_DB_Password": r"postgresql://.*:password@",
    "Hardcoded_API_Key": r'["\']AIzaSy[A-Za-z0-9_-]{40,}["\']',
    "Exposed_Runway_Key": r'["\']key_[a-f0-9]{60,}["\']',
    "API_Key_In_Logs": r"?key=[A-Za-z0-9_-]+",
    "Missing_Env_Variable": r'os\.getenv\(["\']([A-Z_]+)["\'],\s*["\']([^"\']{5,})["\'].*# Default',
}

# Files to exclude from scanning
EXCLUDE_PATTERNS = [
    ".venv311",
    "node_modules",
    ".git",
    "htmlcov",
    "__pycache__",
    "*.pyc",
    ".env.example",
    "SECURITY_AUDIT_FIXES.md",
    "docs/",
]

class SecurityAuditor:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues: List[Tuple[str, str, int, str]] = []
        self.warnings: List[Tuple[str, str, int, str]] = []
        
    def should_scan_file(self, filepath: Path) -> bool:
        """Check if file should be scanned."""
        rel_path = str(filepath.relative_to(self.project_root))
        
        # Skip excluded patterns
        for pattern in EXCLUDE_PATTERNS:
            if pattern in rel_path or filepath.name.endswith(".pyc"):
                return False
        
        # Only scan readable text files
        if filepath.suffix in [".py", ".js", ".json", ".yml", ".yaml", ".env", ".sh", ".md", ".txt"]:
            return True
        
        return False
    
    def scan_file(self, filepath: Path):
        """Scan a single file for secrets."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    self._check_line(filepath, line_num, line)
        except Exception as e:
            print(f"⚠️  Could not scan {filepath}: {e}")
    
    def _check_line(self, filepath: Path, line_num: int, line: str):
        """Check a line for security issues."""
        rel_path = str(filepath.relative_to(self.project_root))
        
        # Check for dangerous patterns (CRITICAL)
        for pattern_name, pattern in DANGEROUS_PATTERNS.items():
            if re.search(pattern, line):
                self.issues.append((rel_path, line_num, pattern_name, line.strip()[:80]))
        
        # Check for secret patterns (WARNING)
        for secret_name, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, line) and "test-key" not in line and "your-" not in line:
                self.warnings.append((rel_path, line_num, secret_name, line.strip()[:80]))
    
    def scan_directory(self):
        """Recursively scan the project directory."""
        print(f"🔍 Scanning {self.project_root} for security issues...\n")
        
        for filepath in self.project_root.rglob("*"):
            if filepath.is_file() and self.should_scan_file(filepath):
                self.scan_file(filepath)
        
        self._print_results()
    
    def _print_results(self):
        """Print audit results."""
        print("\n" + "="*70)
        print("🔐 SECURITY AUDIT RESULTS")
        print("="*70 + "\n")
        
        # Critical Issues
        if self.issues:
            print(f"🔴 CRITICAL ISSUES ({len(self.issues)} found):\n")
            for filepath, line_num, issue_type, line_content in self.issues:
                print(f"  ❌ {filepath}:{line_num}")
                print(f"     Issue: {issue_type}")
                print(f"     Line: {line_content}...")
                print()
        else:
            print("✅ No critical issues found!\n")
        
        # Warnings
        if self.warnings:
            print(f"🟡 WARNINGS ({len(self.warnings)} potential issues):\n")
            for filepath, line_num, warning_type, line_content in self.warnings[:10]:  # Limit output
                print(f"  ⚠️  {filepath}:{line_num}")
                print(f"     Type: {warning_type}")
                print(f"     Line: {line_content}...")
                print()
            
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings\n")
        else:
            print("✅ No warnings found!\n")
        
        # Summary
        print("="*70)
        if self.issues:
            print(f"Summary: ❌ {len(self.issues)} critical issues, 🟡 {len(self.warnings)} warnings")
            return 1
        elif self.warnings:
            print(f"Summary: ✅ No critical issues, 🟡 {len(self.warnings)} warnings to review")
            return 0
        else:
            print("Summary: ✅ All security checks passed!")
            return 0
    
    def export_report(self, filename: str = "security_audit_report.txt"):
        """Export audit results to file."""
        with open(filename, "w") as f:
            f.write("AIPROD Security Audit Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Critical Issues: {len(self.issues)}\n")
            for filepath, line_num, issue_type, line_content in self.issues:
                f.write(f"  {filepath}:{line_num} - {issue_type}\n")
            
            f.write(f"\nWarnings: {len(self.warnings)}\n")
            for filepath, line_num, warning_type, line_content in self.warnings:
                f.write(f"  {filepath}:{line_num} - {warning_type}\n")
        
        print(f"Report saved to {filename}")


def main():
    """Main function."""
    # Use project root
    project_root = os.getenv("AIPROD_HOME", ".")
    
    auditor = SecurityAuditor(project_root)
    auditor.scan_directory()
    exit_code = auditor._print_results()
    
    # Export report
    auditor.export_report("security_audit_report.txt")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

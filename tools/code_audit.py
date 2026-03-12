#!/usr/bin/env python3
"""
Code audit tool for detecting hardcoding patterns that could misrepresent results.

Usage:
    python tools/code_audit.py [path] [--output docs/audit/]
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class AuditIssue:
    line: int
    pattern: str
    code: str
    severity: str  # HIGH, MEDIUM, LOW

@dataclass
class FileAudit:
    file: str
    issues: List[AuditIssue]
    summary: dict

# Detection patterns defined in design doc
PATTERNS = [
    # HIGH severity
    {"name": "magic_assertion", "regex": r"assert\s+.+==\s+\d+\.\d+", "severity": "HIGH"},
    {"name": "mock_import_non_test", "regex": r"^(?:from|import)\s+.*mock", "severity": "HIGH"},
    # MEDIUM severity
    {"name": "hardcoded_threshold", "regex": r"threshold\s*=\s*\d+\.\d+", "severity": "MEDIUM"},
    {"name": "fixture_non_test", "regex": r"@pytest\.fixture", "severity": "MEDIUM"},
    {"name": "paper_value", "regex": r"=\s*(?:0\.683|-0\.7536|0\.5|1\.0)\s*$", "severity": "MEDIUM"},
    {"name": "commented_assertion", "regex": r"#\s*assert|\"\"\"assert", "severity": "MEDIUM"},
    # LOW severity
    {"name": "skipped_test", "regex": r"@pytest\.mark\.(?:skip|xfail)", "severity": "LOW"},
    {"name": "todo_near_result", "regex": r"(?:TODO|FIXME|HACK|XXX)", "severity": "LOW"},
]

SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules"}

def audit_file(filepath: Path) -> FileAudit:
    """Audit a single Python file for hardcoding patterns."""
    issues = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except (UnicodeDecodeError, PermissionError):
        return FileAudit(file=str(filepath), issues=[], summary={"error": "unreadable"})

    # Check if this is a test file (some patterns ignored in tests)
    is_test_file = "test" in filepath.name or "tests" in str(filepath)

    for i, line in enumerate(lines, 1):
        for pattern in PATTERNS:
            # Skip mock import check in test files
            if pattern["name"] == "mock_import_non_test" and is_test_file:
                continue
            # Skip fixture check in test files
            if pattern["name"] == "fixture_non_test" and is_test_file:
                continue

            if re.search(pattern["regex"], line):
                issues.append(AuditIssue(
                    line=i,
                    pattern=pattern["name"],
                    code=line.strip()[:80],
                    severity=pattern["severity"]
                ))

    summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for issue in issues:
        summary[issue.severity] += 1

    return FileAudit(file=str(filepath), issues=issues, summary=summary)

def audit_directory(root: Path, output_dir: Optional[Path] = None) -> List[FileAudit]:
    """Audit all Python files in a directory tree."""
    results = []

    for py_file in root.rglob("*.py"):
        # Skip excluded directories
        if any(skip_dir in py_file.parts for skip_dir in SKIP_DIRS):
            continue
        results.append(audit_file(py_file))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "audit_report.json"
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Audit report written to {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Audit code for hardcoding patterns")
    parser.add_argument("path", nargs="?", default=".", help="Directory to audit")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for reports")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    output_dir = args.output

    print(f"Auditing: {root}")
    results = audit_directory(root, output_dir)

    # Print summary
    total_high = sum(r.summary.get("HIGH", 0) for r in results)
    total_medium = sum(r.summary.get("MEDIUM", 0) for r in results)
    total_low = sum(r.summary.get("LOW", 0) for r in results)

    print(f"\nSummary:")
    print(f"  HIGH:   {total_high}")
    print(f"  MEDIUM: {total_medium}")
    print(f"  LOW:    {total_low}")
    print(f"  Files scanned: {len(results)}")

    # Print files with HIGH issues
    high_files = [r for r in results if r.summary.get("HIGH", 0) > 0]
    if high_files:
        print(f"\nFiles with HIGH severity issues:")
        for f in high_files:
            print(f"  {f.file}")
            for issue in f.issues:
                if issue.severity == "HIGH":
                    print(f"    L{issue.line}: {issue.code}")

if __name__ == "__main__":
    main()
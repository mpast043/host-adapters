"""Tests for code_audit.py"""

import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from code_audit import audit_file, PATTERNS, SKIP_DIRS

def test_detects_magic_assertion():
    """Should detect magic number assertions."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("assert result == 0.683\n")
        f.flush()
        audit = audit_file(Path(f.name))
        assert any(i.pattern == "magic_assertion" for i in audit.issues)
        assert audit.summary["HIGH"] >= 1

def test_detects_hardcoded_threshold():
    """Should detect hardcoded thresholds."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("threshold = 0.95\n")
        f.flush()
        audit = audit_file(Path(f.name))
        assert any(i.pattern == "hardcoded_threshold" for i in audit.issues)

def test_skips_venv():
    """Should skip virtualenv directories."""
    assert "venv" in SKIP_DIRS
    assert ".venv" in SKIP_DIRS

def test_ignores_mock_in_test_file():
    """Should not flag mock imports in test files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:
        f.write("from unittest.mock import Mock\n")
        f.flush()
        audit = audit_file(Path(f.name))
        assert not any(i.pattern == "mock_import_non_test" for i in audit.issues)

def test_detects_paper_value():
    """Should detect paper reference values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("expected = 0.683\n")
        f.flush()
        audit = audit_file(Path(f.name))
        assert any(i.pattern == "paper_value" for i in audit.issues)

def test_detects_commented_assertion():
    """Should detect commented out assertions."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# assert result == expected\n")
        f.flush()
        audit = audit_file(Path(f.name))
        assert any(i.pattern == "commented_assertion" for i in audit.issues)

def test_detects_todo_near_result():
    """Should detect TODO/FIXME near results."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# TODO: fix this later\nresult = compute()\n")
        f.flush()
        audit = audit_file(Path(f.name))
        assert any(i.pattern == "todo_near_result" for i in audit.issues)
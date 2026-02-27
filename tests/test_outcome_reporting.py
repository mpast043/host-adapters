"""
tests/test_outcome_reporting.py

Unit tests for outcome reporting: verifies that double-failures (HTTP + JSONL)
surface as GovernanceError rather than being silently dropped.
"""

import asyncio
import json
import sys
import builtins
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make SDK importable
_SDK_PATH = Path(__file__).parent.parent / "sdk" / "python"
if str(_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(_SDK_PATH))

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "adapters"))

from cgf_sdk.errors import GovernanceError


# ---------------------------------------------------------------------------
# OpenClaw adapter tests
# ---------------------------------------------------------------------------

class TestOpenClawOutcomeReporting:
    """Verify openclaw_adapter_v02 report_outcome() never silently drops."""

    def _make_adapter(self, tmp_path):
        from openclaw_adapter_v02 import OpenClawAdapter
        adapter = OpenClawAdapter(
            cgf_endpoint="http://127.0.0.1:9999",  # unreachable
            data_dir=tmp_path
        )
        adapter.adapter_id = "test-adapter"
        # Provide a mock aiohttp session
        mock_session = MagicMock()
        # session.post() returns an async context manager
        ctx_manager = AsyncMock()
        ctx_manager.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        ctx_manager.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=ctx_manager)
        adapter.session = mock_session
        return adapter

    def _make_proposal(self):
        from cgf_schemas_v02 import HostProposal, ActionType, RiskTier
        return HostProposal(
            proposal_id="prop-test-001",
            timestamp=0.0,
            action_type=ActionType.TOOL_CALL,
            action_params={"tool_name": "ls"},
            context_refs=[],
            estimated_cost={},
            risk_tier=RiskTier.LOW,
            priority=0,
        )

    @pytest.mark.asyncio
    async def test_http_failure_writes_local_jsonl(self, tmp_path):
        """When HTTP fails, outcome is written to local JSONL."""
        adapter = self._make_adapter(tmp_path)
        proposal = self._make_proposal()

        await adapter.report_outcome(
            proposal=proposal,
            decision_id="dec-001",
            executed=True,
            success=True,
            committed=True,
            quarantined=False,
            duration_ms=5.0,
        )

        local_file = tmp_path / "outcomes_local.jsonl"
        assert local_file.exists(), "Local JSONL fallback must be written on HTTP failure"
        lines = [l for l in local_file.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "report_error" in record
        assert record["proposal_id"] == "prop-test-001"

    @pytest.mark.asyncio
    async def test_double_failure_raises_governance_error(self, tmp_path):
        """When both HTTP and JSONL fail, GovernanceError is raised (no silent drop)."""
        adapter = self._make_adapter(tmp_path)
        proposal = self._make_proposal()

        # Patch open() so that writing to outcomes_local.jsonl fails
        _real_open = builtins.open

        def _failing_open(path, *args, **kwargs):
            if "outcomes_local" in str(path):
                raise OSError("simulated disk full")
            return _real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_failing_open):
            with pytest.raises(GovernanceError) as exc_info:
                await adapter.report_outcome(
                    proposal=proposal,
                    decision_id="dec-002",
                    executed=True,
                    success=True,
                    committed=True,
                    quarantined=False,
                    duration_ms=5.0,
                )

        assert exc_info.value.error_code == "OUTCOME_LOSS"
        assert exc_info.value.proposal_id == "prop-test-001"


# ---------------------------------------------------------------------------
# LangGraph adapter tests
# ---------------------------------------------------------------------------

class TestLangGraphOutcomeReporting:
    """Verify langgraph_adapter_v01 report() never silently drops."""

    def _make_outcome(self):
        try:
            from cgf_schemas_v03 import HostOutcomeReport
        except ImportError:
            from cgf_schemas_v02 import HostOutcomeReport
        return HostOutcomeReport(
            adapter_id="test-lg-adapter",
            proposal_id="lg-prop-001",
            decision_id="lg-dec-001",
            executed=True,
            executed_at=0.0,
            duration_ms=10.0,
            success=True,
            committed=True,
            quarantined=False,
            errors=[],
            result_summary="OK",
        )

    @pytest.mark.asyncio
    async def test_http_failure_writes_local_jsonl(self, tmp_path):
        """When HTTP fails, outcome is written to local JSONL."""
        from langgraph_adapter_v01 import LangGraphAdapter, DEFAULT_CONFIG
        adapter = LangGraphAdapter()
        adapter.adapter_id = "test-lg"

        outcome = self._make_outcome()

        original_dir = DEFAULT_CONFIG["data_dir"]
        DEFAULT_CONFIG["data_dir"] = str(tmp_path)
        try:
            with patch("aiohttp.ClientSession") as mock_cls:
                mock_session = AsyncMock()
                mock_post_cm = AsyncMock()
                mock_post_cm.__aenter__ = AsyncMock(side_effect=Exception("network down"))
                mock_post_cm.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_post_cm)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_session

                await adapter.report(outcome)
        finally:
            DEFAULT_CONFIG["data_dir"] = original_dir

        local_file = tmp_path / "outcomes.jsonl"
        assert local_file.exists()
        lines = [l for l in local_file.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record.get("proposal_id") == "lg-prop-001"
        assert "http_error" in record

    @pytest.mark.asyncio
    async def test_double_failure_raises_governance_error(self, tmp_path):
        """When both HTTP and JSONL fail, GovernanceError is raised."""
        from langgraph_adapter_v01 import LangGraphAdapter, DEFAULT_CONFIG
        adapter = LangGraphAdapter()
        adapter.adapter_id = "test-lg"

        outcome = self._make_outcome()

        original_dir = DEFAULT_CONFIG["data_dir"]
        DEFAULT_CONFIG["data_dir"] = str(tmp_path)

        _real_open = builtins.open

        def _failing_open(path, *args, **kwargs):
            if "outcomes.jsonl" in str(path):
                raise OSError("simulated disk full")
            return _real_open(path, *args, **kwargs)

        try:
            with patch("aiohttp.ClientSession") as mock_cls:
                mock_session = AsyncMock()
                mock_post_cm = AsyncMock()
                mock_post_cm.__aenter__ = AsyncMock(side_effect=Exception("network down"))
                mock_post_cm.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_post_cm)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_session

                with patch("builtins.open", side_effect=_failing_open):
                    with pytest.raises(GovernanceError) as exc_info:
                        await adapter.report(outcome)
        finally:
            DEFAULT_CONFIG["data_dir"] = original_dir

        assert exc_info.value.error_code == "OUTCOME_LOSS"
        assert exc_info.value.proposal_id == "lg-prop-001"

"""
tests/test_circuit_breaker.py

Unit tests for the CircuitBreaker inside cgf_client.py.

Tests use the CircuitBreaker class directly (no HTTP) and also verify that
CGFClient honours circuit-breaker state via mocked aiohttp sessions.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# Make SDK importable
_SDK_PATH = Path(__file__).parent.parent / "sdk" / "python"
if str(_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(_SDK_PATH))

from cgf_sdk.cgf_client import CircuitBreaker, CGFClient, ClientConfig
from cgf_sdk.errors import CGFConnectionError


# ---------------------------------------------------------------------------
# CircuitBreaker unit tests (no HTTP)
# ---------------------------------------------------------------------------

class TestCircuitBreakerStates:

    def _breaker(self, threshold=3, cooldown_s=60.0, half_open_max=1):
        return CircuitBreaker(failure_threshold=threshold, cooldown_s=cooldown_s, half_open_max=half_open_max)

    def test_initial_state_is_closed(self):
        cb = self._breaker()
        assert cb.state == "closed"

    def test_closed_check_does_not_raise(self):
        cb = self._breaker()
        cb.check()  # should not raise

    def test_success_resets_failure_count(self):
        cb = self._breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Only 2 failures after the reset, still below threshold of 3
        assert cb.state == "closed"

    def test_failures_below_threshold_stay_closed(self):
        cb = self._breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_threshold_failures_open_circuit(self):
        cb = self._breaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_open_check_raises_circuit_open(self):
        cb = self._breaker(threshold=1, cooldown_s=9999.0)
        cb.record_failure()
        assert cb.state == "open"
        with pytest.raises(CGFConnectionError) as exc_info:
            cb.check()
        assert exc_info.value.error_code == "CIRCUIT_OPEN"

    def test_open_transitions_to_half_open_after_cooldown(self):
        cb = self._breaker(threshold=1, cooldown_s=0.0)
        cb.record_failure()
        assert cb.state == "open"
        # Cooldown is 0 so next check should transition to HALF_OPEN
        cb.check()  # should not raise — transitions to half_open and counts probe
        assert cb.state == "half_open"

    def test_half_open_success_closes_circuit(self):
        cb = self._breaker(threshold=1, cooldown_s=0.0, half_open_max=1)
        cb.record_failure()
        cb.check()  # → HALF_OPEN, probe #1 allowed
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens_circuit(self):
        cb = self._breaker(threshold=1, cooldown_s=0.0, half_open_max=1)
        cb.record_failure()
        cb.check()  # → HALF_OPEN, probe allowed
        cb.record_failure()  # probe failed → OPEN again
        assert cb.state == "open"

    def test_half_open_probe_limit_raises(self):
        cb = self._breaker(threshold=1, cooldown_s=0.0, half_open_max=1)
        cb.record_failure()
        cb.check()  # → HALF_OPEN, probe #1 used up
        # Second check in HALF_OPEN should raise (probe limit reached)
        with pytest.raises(CGFConnectionError) as exc_info:
            cb.check()
        assert exc_info.value.error_code == "CIRCUIT_OPEN"


# ---------------------------------------------------------------------------
# CGFClient integration tests
# ---------------------------------------------------------------------------

def _make_client_with_breaker(threshold=3, cooldown_ms=60000, half_open_max=1):
    """Create a CGFClient with a fresh circuit breaker."""
    client = CGFClient(config=ClientConfig(endpoint="http://127.0.0.1:19999", timeout_ms=100))
    client._breaker = CircuitBreaker(threshold, cooldown_ms / 1000, half_open_max)
    return client


def _make_failing_session():
    """aiohttp session mock that raises aiohttp.ClientError on POST context-enter."""
    mock_session = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection refused"))
    ctx.__aexit__ = AsyncMock(return_value=None)
    mock_session.post = MagicMock(return_value=ctx)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


class TestCGFClientCircuitBreaker:

    @pytest.mark.asyncio
    async def test_disabled_breaker_does_not_open(self):
        """When _breaker is None (disabled), failures never trigger CIRCUIT_OPEN."""
        client = CGFClient(config=ClientConfig(endpoint="http://127.0.0.1:19999", timeout_ms=100))
        assert client._breaker is None  # disabled by default

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = _make_failing_session()
            for _ in range(5):
                with pytest.raises(CGFConnectionError) as exc_info:
                    await client.evaluate_async({}, {})
                # Should always be the network error, never CIRCUIT_OPEN
                assert exc_info.value.error_code != "CIRCUIT_OPEN"

    @pytest.mark.asyncio
    async def test_open_circuit_raises_without_network_call(self):
        """When circuit is OPEN, evaluate_async raises before any network call."""
        client = _make_client_with_breaker(threshold=1, cooldown_ms=60000)
        # Force circuit open
        client._breaker.record_failure()
        assert client._breaker.state == "open"

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(CGFConnectionError) as exc_info:
                await client.evaluate_async({}, {})

        assert exc_info.value.error_code == "CIRCUIT_OPEN"
        # No HTTP call should have been attempted
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_threshold_failures_open_circuit(self):
        """Three consecutive network failures open the circuit."""
        client = _make_client_with_breaker(threshold=3, cooldown_ms=60000)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = _make_failing_session()
            for _ in range(3):
                with pytest.raises(CGFConnectionError):
                    await client.evaluate_async({}, {})

        assert client._breaker.state == "open"

    @pytest.mark.asyncio
    async def test_success_resets_breaker(self):
        """A successful call resets the circuit to CLOSED."""
        client = _make_client_with_breaker(threshold=3, cooldown_ms=0)

        # Build a successful session mock
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"decision": "ALLOW"})
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Add two failures first
        client._breaker.record_failure()
        client._breaker.record_failure()
        assert client._breaker.state == "closed"

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client.evaluate_async({}, {})

        assert client._breaker.state == "closed"
        assert client._breaker._failures == 0

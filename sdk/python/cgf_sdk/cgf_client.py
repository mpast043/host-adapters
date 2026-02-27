"""
sdk/python/cgf_sdk/cgf_client.py

Typed client for CGF REST API.
Provides synchronous and async interfaces for adapter registration,
proposal evaluation, and outcome reporting.
"""

import asyncio
import enum
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .errors import CGFConnectionError, CGFEvaluationError, CGFRegistryError


# ============== CIRCUIT BREAKER ==============

class _CBState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Three-state circuit breaker for CGF client calls.

    CLOSED  → normal operation; failures increment counter.
    OPEN    → CGF is considered down; calls raise CGFConnectionError(CIRCUIT_OPEN)
              immediately without hitting the network.
    HALF_OPEN → one probe call is allowed after cooldown; success resets to
                CLOSED, failure returns to OPEN.
    """

    def __init__(self, failure_threshold: int, cooldown_s: float, half_open_max: int):
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._half_open_max = half_open_max
        self._state = _CBState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state.value

    def check(self) -> None:
        """Raise CGFConnectionError(CIRCUIT_OPEN) when the circuit is open."""
        with self._lock:
            if self._state == _CBState.CLOSED:
                return
            if self._state == _CBState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._cooldown_s:
                    self._state = _CBState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    raise CGFConnectionError(
                        "Circuit breaker open — CGF unreachable",
                        error_code="CIRCUIT_OPEN",
                    )
            # HALF_OPEN: gate to at most half_open_max probe calls
            if self._state == _CBState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max:
                    raise CGFConnectionError(
                        "Circuit breaker half-open — probe limit reached",
                        error_code="CIRCUIT_OPEN",
                    )
                self._half_open_calls += 1

    def record_success(self) -> None:
        """Reset the breaker to CLOSED after a successful call."""
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._state = _CBState.CLOSED

    def record_failure(self) -> None:
        """Record a failure; open the circuit when threshold is reached."""
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold or self._state == _CBState.HALF_OPEN:
                self._state = _CBState.OPEN
                self._opened_at = time.monotonic()


# Circuit-breaker env-var config (read once at import time so tests can monkeypatch)
_CB_ENABLED = os.environ.get("CGF_CIRCUIT_BREAKER", "0") == "1"
_CB_THRESHOLD = int(os.environ.get("CGF_CB_FAILURE_THRESHOLD", "3"))
_CB_COOLDOWN_MS = int(os.environ.get("CGF_CB_COOLDOWN_MS", "2000"))
_CB_HALF_OPEN = int(os.environ.get("CGF_CB_HALF_OPEN_MAX_CALLS", "1"))


@dataclass
class ClientConfig:
    """Configuration for CGF client."""
    endpoint: str = "http://127.0.0.1:8080"
    timeout_ms: int = 500
    retry_count: int = 2
    retry_delay_ms: int = 100
    api_version: str = "v1"
    schema_version: str = "0.3.0"
    
    @property
    def base_url(self) -> str:
        return urljoin(self.endpoint, f"/{self.api_version}")


class CGFClient:
    """
    Typed client for Capacity Governance Framework.
    
    Synchronizes with cgf_schemas_v03.py types while providing
    clean Python interface.
    """
    
    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()
        self.session: Optional[Any] = None
        self._adapter_id: Optional[str] = None
        self._breaker: Optional[CircuitBreaker] = (
            CircuitBreaker(_CB_THRESHOLD, _CB_COOLDOWN_MS / 1000, _CB_HALF_OPEN)
            if _CB_ENABLED else None
        )
        
    # ============ Registration ============
    
    async def register_async(
        self,
        adapter_type: str,
        capabilities: List[str],
        host_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Async registration with CGF."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for async operations")
        
        payload = {
            "schema_version": self.config.schema_version,
            "adapter_type": adapter_type,
            "capabilities": capabilities,
            "host_config": host_config,
            "supported_actions": capabilities
        }
        
        url = f"{self.config.base_url}/adapters/register"

        if self._breaker:
            self._breaker.check()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000)
                ) as resp:
                    if resp.status != 200:
                        if self._breaker:
                            self._breaker.record_failure()
                        raise CGFRegistryError(
                            f"Registration failed: {resp.status}",
                            status_code=resp.status
                        )
                    data = await resp.json()
                    self._adapter_id = data.get("adapter_id")
                    if self._breaker:
                        self._breaker.record_success()
                    return data
            except asyncio.TimeoutError as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(
                    f"Registration timeout after {self.config.timeout_ms}ms"
                ) from e
            except aiohttp.ClientError as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    def register(
        self,
        adapter_type: str,
        capabilities: List[str],
        host_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous registration."""
        if HAS_AIOHTTP:
            return asyncio.run(self.register_async(adapter_type, capabilities, host_config))
        elif HAS_REQUESTS:
            return self._register_sync_requests(adapter_type, capabilities, host_config)
        else:
            raise ImportError("Either aiohttp or requests required")
    
    def _register_sync_requests(
        self,
        adapter_type: str,
        capabilities: List[str],
        host_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous registration using requests."""
        import requests
        
        payload = {
            "schema_version": self.config.schema_version,
            "adapter_type": adapter_type,
            "capabilities": capabilities,
            "host_config": host_config,
            "supported_actions": capabilities
        }
        
        url = f"{self.config.base_url}/adapters/register"

        if self._breaker:
            self._breaker.check()

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_ms / 1000
            )
            resp.raise_for_status()
            data = resp.json()
            self._adapter_id = data.get("adapter_id")
            if self._breaker:
                self._breaker.record_success()
            return data
        except requests.Timeout as e:
            if self._breaker:
                self._breaker.record_failure()
            raise CGFConnectionError(
                f"Registration timeout after {self.config.timeout_ms}ms"
            ) from e
        except requests.RequestException as e:
            if self._breaker:
                self._breaker.record_failure()
            raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    # ============ Evaluation ============
    
    async def evaluate_async(
        self,
        proposal: Dict[str, Any],
        context: Dict[str, Any],
        capacity_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async evaluate proposal."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for async operations")
        
        payload = {
            "schema_version": self.config.schema_version,
            "proposal": proposal,
            "context": context,
            "capacity_signals": capacity_signals or self._default_capacity(),
            "observed_at": time.time()
        }
        
        url = f"{self.config.base_url}/evaluate"

        if self._breaker:
            self._breaker.check()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000)
                ) as resp:
                    if resp.status != 200:
                        if self._breaker:
                            self._breaker.record_failure()
                        raise CGFEvaluationError(
                            f"Evaluation failed: {resp.status}",
                            status_code=resp.status
                        )
                    data = await resp.json()
                    if self._breaker:
                        self._breaker.record_success()
                    return data
            except asyncio.TimeoutError as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(
                    f"Evaluation timeout after {self.config.timeout_ms}ms"
                ) from e
            except aiohttp.ClientError as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    def evaluate(
        self,
        proposal: Dict[str, Any],
        context: Dict[str, Any],
        capacity_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous evaluation."""
        if HAS_AIOHTTP:
            return asyncio.run(self.evaluate_async(proposal, context, capacity_signals))
        elif HAS_REQUESTS:
            return self._evaluate_sync_requests(proposal, context, capacity_signals)
        else:
            raise ImportError("Either aiohttp or requests required")
    
    def _evaluate_sync_requests(
        self,
        proposal: Dict[str, Any],
        context: Dict[str, Any],
        capacity_signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous evaluation using requests."""
        import requests
        
        payload = {
            "schema_version": self.config.schema_version,
            "proposal": proposal,
            "context": context,
            "capacity_signals": capacity_signals or self._default_capacity(),
            "observed_at": time.time()
        }
        
        url = f"{self.config.base_url}/evaluate"

        if self._breaker:
            self._breaker.check()

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_ms / 1000
            )
            resp.raise_for_status()
            data = resp.json()
            if self._breaker:
                self._breaker.record_success()
            return data
        except requests.Timeout as e:
            if self._breaker:
                self._breaker.record_failure()
            raise CGFConnectionError(
                f"Evaluation timeout after {self.config.timeout_ms}ms"
            ) from e
        except requests.RequestException as e:
            if self._breaker:
                self._breaker.record_failure()
            raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    # ============ Outcome Reporting ============
    
    async def report_outcome_async(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Async report outcome."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for async operations")
        
        url = f"{self.config.base_url}/outcomes/report"

        if self._breaker:
            self._breaker.check()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=outcome,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000)
                ) as resp:
                    if resp.status != 200:
                        if self._breaker:
                            self._breaker.record_failure()
                        raise CGFEvaluationError(
                            f"Outcome report failed: {resp.status}",
                            status_code=resp.status
                        )
                    data = await resp.json()
                    if self._breaker:
                        self._breaker.record_success()
                    return data
            except asyncio.TimeoutError as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(f"Outcome timeout") from e
            except aiohttp.ClientError as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    def report_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous outcome report."""
        if HAS_AIOHTTP:
            return asyncio.run(self.report_outcome_async(outcome))
        elif HAS_REQUESTS:
            import requests

            if self._breaker:
                self._breaker.check()

            url = f"{self.config.base_url}/outcomes/report"
            try:
                resp = requests.post(url, json=outcome, timeout=self.config.timeout_ms / 1000)
                resp.raise_for_status()
                data = resp.json()
                if self._breaker:
                    self._breaker.record_success()
                return data
            except requests.RequestException as e:
                if self._breaker:
                    self._breaker.record_failure()
                raise CGFConnectionError(f"CGF connection failed: {e}") from e
        else:
            raise ImportError("Either aiohttp or requests required")
    
    # ============ Health Check ============
    
    async def health_async(self) -> Dict[str, Any]:
        """Async health check."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")
        
        url = urljoin(self.config.endpoint, "/health")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return await resp.json()
            except Exception:
                return {"status": "unreachable"}
    
    def health(self) -> Dict[str, Any]:
        """Synchronous health check."""
        if HAS_AIOHTTP:
            return asyncio.run(self.health_async())
        elif HAS_REQUESTS:
            import requests
            try:
                url = urljoin(self.config.endpoint, "/health")
                resp = requests.get(url, timeout=5)
                return resp.json()
            except Exception:
                return {"status": "unreachable"}
        return {"status": "no_http_client"}
    
    # ============ Helpers ============
    
    @property
    def adapter_id(self) -> Optional[str]:
        return self._adapter_id
    
    def _default_capacity(self) -> Dict[str, float]:
        """Default capacity signals when not provided."""
        return {
            "C_geo_available": 0.95,
            "C_geo_total": 1.0,
            "C_int_available": 0.95,
            "C_gauge_available": 0.95,
            "C_ptr_available": 0.95,
            "C_obs_available": 0.95,
            "gate_fit_margin": 0.2,
            "gate_gluing_margin": 0.15
        }

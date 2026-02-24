"""
sdk/python/cgf_sdk/cgf_client.py

Typed client for CGF REST API.
Provides synchronous and async interfaces for adapter registration,
proposal evaluation, and outcome reporting.
"""

import asyncio
import json
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
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000)
                ) as resp:
                    if resp.status != 200:
                        raise CGFRegistryError(
                            f"Registration failed: {resp.status}",
                            status_code=resp.status
                        )
                    data = await resp.json()
                    self._adapter_id = data.get("adapter_id")
                    return data
            except asyncio.TimeoutError as e:
                raise CGFConnectionError(
                    f"Registration timeout after {self.config.timeout_ms}ms"
                ) from e
            except aiohttp.ClientError as e:
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
        
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_ms / 1000
            )
            resp.raise_for_status()
            data = resp.json()
            self._adapter_id = data.get("adapter_id")
            return data
        except requests.Timeout as e:
            raise CGFConnectionError(
                f"Registration timeout after {self.config.timeout_ms}ms"
            ) from e
        except requests.RequestException as e:
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
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000)
                ) as resp:
                    if resp.status != 200:
                        raise CGFEvaluationError(
                            f"Evaluation failed: {resp.status}",
                            status_code=resp.status
                        )
                    return await resp.json()
            except asyncio.TimeoutError as e:
                raise CGFConnectionError(
                    f"Evaluation timeout after {self.config.timeout_ms}ms"
                ) from e
            except aiohttp.ClientError as e:
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
        
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_ms / 1000
            )
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as e:
            raise CGFConnectionError(
                f"Evaluation timeout after {self.config.timeout_ms}ms"
            ) from e
        except requests.RequestException as e:
            raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    # ============ Outcome Reporting ============
    
    async def report_outcome_async(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Async report outcome."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for async operations")
        
        url = f"{self.config.base_url}/outcomes/report"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=outcome,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000)
                ) as resp:
                    if resp.status != 200:
                        raise CGFEvaluationError(
                            f"Outcome report failed: {resp.status}",
                            status_code=resp.status
                        )
                    return await resp.json()
            except asyncio.TimeoutError as e:
                raise CGFConnectionError(f"Outcome timeout") from e
            except aiohttp.ClientError as e:
                raise CGFConnectionError(f"CGF connection failed: {e}") from e
    
    def report_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous outcome report."""
        if HAS_AIOHTTP:
            return asyncio.run(self.report_outcome_async(outcome))
        elif HAS_REQUESTS:
            import requests
            
            url = f"{self.config.base_url}/outcomes/report"
            try:
                resp = requests.post(url, json=outcome, timeout=self.config.timeout_ms / 1000)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
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

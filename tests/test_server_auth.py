"""
tests/test_server_auth.py

Tests for optional bearer-token authentication on CGF server write endpoints.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make server importable
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

import cgf_server_v03 as _server_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_with_token(token: str) -> TestClient:
    """Return a TestClient that uses app with CGF_AUTH_TOKEN patched to *token*."""
    _server_module.CGF_AUTH_TOKEN = token
    return TestClient(_server_module.app)


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# When CGF_AUTH_TOKEN is empty (default) — no auth required
# ---------------------------------------------------------------------------

class TestAuthDisabled:
    """When CGF_AUTH_TOKEN is empty, all requests pass regardless of header."""

    def setup_method(self):
        _server_module.CGF_AUTH_TOKEN = ""
        self.client = TestClient(_server_module.app)

    def test_evaluate_no_header_not_401(self):
        """With auth disabled, missing header should not produce 401."""
        resp = self.client.post("/v1/evaluate", json={})
        assert resp.status_code != 401

    def test_register_no_header_not_401(self):
        resp = self.client.post("/v1/register", json={})
        assert resp.status_code != 401

    def test_outcomes_no_header_not_401(self):
        resp = self.client.post("/v1/outcomes/report", json={})
        assert resp.status_code != 401

    def test_health_always_accessible(self):
        resp = self.client.get("/v1/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# When CGF_AUTH_TOKEN is set — auth is enforced
# ---------------------------------------------------------------------------

SECRET = "test-secret-token-abc123"


class TestAuthEnabled:
    """When CGF_AUTH_TOKEN is set, write endpoints require the matching token."""

    def setup_method(self):
        self.client = _client_with_token(SECRET)

    def teardown_method(self):
        _server_module.CGF_AUTH_TOKEN = ""

    # ---- /v1/evaluate ----

    def test_evaluate_correct_token_not_401(self):
        """Correct token → not 401 (may be 422 for bad body, but auth passes)."""
        resp = self.client.post("/v1/evaluate", json={}, headers=_auth_header(SECRET))
        assert resp.status_code != 401

    def test_evaluate_wrong_token_is_401(self):
        resp = self.client.post("/v1/evaluate", json={}, headers=_auth_header("wrong"))
        assert resp.status_code == 401

    def test_evaluate_missing_header_is_401(self):
        resp = self.client.post("/v1/evaluate", json={})
        assert resp.status_code == 401

    def test_evaluate_malformed_header_is_401(self):
        """A header without 'Bearer ' prefix is rejected."""
        resp = self.client.post(
            "/v1/evaluate", json={},
            headers={"Authorization": SECRET}  # missing "Bearer " prefix
        )
        assert resp.status_code == 401

    # ---- /v1/register ----

    def test_register_correct_token_not_401(self):
        resp = self.client.post("/v1/register", json={}, headers=_auth_header(SECRET))
        assert resp.status_code != 401

    def test_register_wrong_token_is_401(self):
        resp = self.client.post("/v1/register", json={}, headers=_auth_header("bad"))
        assert resp.status_code == 401

    def test_register_missing_header_is_401(self):
        resp = self.client.post("/v1/register", json={})
        assert resp.status_code == 401

    # ---- /v1/outcomes/report ----

    def test_outcomes_correct_token_not_401(self):
        resp = self.client.post("/v1/outcomes/report", json={}, headers=_auth_header(SECRET))
        assert resp.status_code != 401

    def test_outcomes_wrong_token_is_401(self):
        resp = self.client.post("/v1/outcomes/report", json={}, headers=_auth_header("nope"))
        assert resp.status_code == 401

    def test_outcomes_missing_header_is_401(self):
        resp = self.client.post("/v1/outcomes/report", json={})
        assert resp.status_code == 401

    # ---- /v1/health is always accessible ----

    def test_health_accessible_without_token(self):
        """Health check must never require auth."""
        resp = self.client.get("/v1/health")
        assert resp.status_code == 200

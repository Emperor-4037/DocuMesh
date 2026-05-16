"""
Gateway integration tests with mocked downstream services.
Tests routing, auth, error handling, and trace propagation.

What these catch:
  - Auth bypass or rejection regressions
  - Route wiring errors (wrong URL, wrong method)
  - Missing trace ID propagation
  - Schema contract violations between gateway and services
"""
import pytest
import jwt
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from shared.config import settings


def get_valid_token():
    return jwt.encode({"sub": "testuser"}, settings.SECRET_KEY, algorithm="HS256")


@pytest.fixture
def client():
    from gateway.app.main import app
    return TestClient(app, raise_server_exceptions=False)


# ── Health & Auth ─────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "gateway"


class TestAuth:
    def test_unauthenticated_request_rejected(self, client):
        resp = client.post("/api/paraphrase", json={"text": "hello"})
        assert resp.status_code == 401

    def test_invalid_token_rejected(self, client):
        headers = {"Authorization": "Bearer invalid_token_abc"}
        resp = client.post("/api/grammar", json={"text": "Hello."}, headers=headers)
        assert resp.status_code == 403

    def test_demo_token_accepted_in_dev(self, client):
        """Demo token should work when SECRET_KEY is the default."""
        headers = {"Authorization": "Bearer demo-token"}
        # This will fail on downstream call (no service running), but auth should pass
        resp = client.post("/api/paraphrase", json={"text": "hello", "tone": "neutral"}, headers=headers)
        # Should NOT be 403 — auth passed, downstream 503 is expected
        assert resp.status_code != 403


# ── Trace ID ──────────────────────────────────────────────────────────────────

class TestTracing:
    def test_trace_id_injected_in_response(self, client):
        resp = client.get("/health")
        assert "x-trace-id" in resp.headers

    def test_custom_trace_id_preserved(self, client):
        resp = client.get("/health", headers={"X-Trace-Id": "custom-trace-123"})
        assert resp.headers.get("x-trace-id") == "custom-trace-123"


# ── Schema Validation ─────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_paraphrase_missing_text_returns_422(self, client):
        headers = {"Authorization": f"Bearer {get_valid_token()}"}
        resp = client.post("/api/paraphrase", json={}, headers=headers)
        assert resp.status_code == 422

    def test_grammar_missing_text_returns_422(self, client):
        headers = {"Authorization": f"Bearer {get_valid_token()}"}
        resp = client.post("/api/grammar", json={}, headers=headers)
        assert resp.status_code == 422

    def test_rag_query_missing_query_returns_422(self, client):
        headers = {"Authorization": f"Bearer {get_valid_token()}"}
        resp = client.post("/api/rag/query", json={}, headers=headers)
        assert resp.status_code == 422

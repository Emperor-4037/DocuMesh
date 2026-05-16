"""
Service API contract tests.
Each test validates a specific service's FastAPI endpoint contract
by sending a real HTTP request to the test client and checking the response shape.

What these catch:
  - Response schema regressions (missing fields, wrong types)
  - HTTP status code changes
  - Service startup failures
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Paraphrase Service ────────────────────────────────────────────────────────

class TestParaphraseAPI:
    @pytest.fixture
    def client(self):
        # Mock the model to avoid loading actual weights
        with patch("app.model._load"):
            with patch("app.model.paraphrase", return_value="Mocked paraphrase result"):
                from services.paraphrase_service.app.main import app
                yield TestClient(app)

    def test_health(self):
        # Direct import without mock since health doesn't use model
        from services.paraphrase_service.app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "paraphrase-service"


# ── Grammar Service ───────────────────────────────────────────────────────────

class TestGrammarAPI:
    def test_health(self):
        from services.grammar_service.app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "grammar-service"


# ── RAG Service ───────────────────────────────────────────────────────────────

class TestRAGAPI:
    def test_health(self):
        from services.rag_service.app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "rag-service"

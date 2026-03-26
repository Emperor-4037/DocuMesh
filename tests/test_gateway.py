import pytest
import jwt
from fastapi.testclient import TestClient
from gateway.app.main import app
from shared.config import settings

client = TestClient(app)

def get_valid_token():
    return jwt.encode({"sub": "testuser"}, settings.SECRET_KEY, algorithm="HS256")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "gateway"}

def test_auth_rejection():
    # Attempting to hit a protected route without a token
    response = client.post("/api/paraphrase", json={"text": "hello", "tone": "neutral"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authenticated"

def test_invalid_token():
    headers = {"Authorization": "Bearer invalid_token_abc"}
    response = client.post("/api/grammar", json={"text": "Hello world!"}, headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Could not validate credentials"

# To test the downstream routes fully, one would normally mock the `httpx.AsyncClient` 
# or run the full docker-compose suite for integration testing. These tests ensure
# that the Gateway correctly intercepts and secures the traffic.

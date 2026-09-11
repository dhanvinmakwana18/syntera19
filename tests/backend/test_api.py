import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["ONLINE", "DEGRADED", "OFFLINE"]

def test_status_endpoint():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert "operational" in response.json()["status"]

def test_chat_direct_mode():
    payload = {"query": "Hello", "mode": "direct"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["trace"]) > 0
    assert any("ROUTER" in step["step"] for step in data["trace"])

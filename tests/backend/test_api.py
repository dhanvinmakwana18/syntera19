import pytest
from fastapi.testclient import TestClient
from api.main import app
from core.container import build_container

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["ONLINE", "DEGRADED", "OFFLINE"]

def test_status_endpoint(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert "operational" in response.json()["status"]

def test_chat_direct_mode(client):
    class FakeLLM:
        def generate(self, *args, **kwargs):
            return "Fake response"
    client.app.state.container._instances["llm"] = FakeLLM()
    
    payload = {"query": "Hello", "mode": "direct"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["trace"]) > 0
    assert any("ROUTER" in step["step"] for step in data["trace"])

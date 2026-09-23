import pytest
from fastapi.testclient import TestClient
from api.main import app
from core.container import build_container

@pytest.fixture
def client():
    # Use context manager to trigger lifespan
    with TestClient(app) as c:
        yield c

def test_empty_retrieval_insufficient_evidence(client):
    response = client.post("/api/v1/chat", json={
        "query": "What is the exact name of the fictional alien species in this nonexistent document?",
        "mode": "rag"
    })
    assert response.status_code == 200
    data = response.json()
    assert "I cannot find sufficient evidence" in data["answer"] or not data["grounded"]

def test_vector_store_failure(client):
    # Hijack container
    class BrokenStore:
        def retrieve(self, *args, **kwargs):
            raise Exception("VECTOR_STORE_FAILURE")
            
    # Modify the container on app.state
    client.app.state.container._instances["vector_store"] = BrokenStore()
    client.app.state.container._instances["bm25_store"] = BrokenStore()

    response = client.post("/api/v1/chat", json={
        "query": "test query",
        "mode": "rag"
    })
    assert response.status_code == 200
    data = response.json()
    assert "Retrieval error:" in data["answer"] or "VECTOR_STORE_FAILURE" in data["answer"]

def test_model_failure(client):
    class BrokenLLM:
        def generate(self, *args, **kwargs):
            raise Exception("MODEL_FAILURE")
            
    client.app.state.container._instances["intelligence"] = BrokenLLM()
    response = client.post("/api/v1/chat", json={
        "query": "test query",
        "mode": "direct"
    })
    assert response.status_code == 200
    data = response.json()
    assert "Error:" in data["answer"] or "MODEL_FAILURE" in data["answer"]

def test_invalid_citation():
    from verification.grounding import validate_citations
    hallucinated_response = "Here is an answer [Source 99]."
    valid_sources = [{"id": "1", "filename": "test.txt", "text": "content"}]
    validated = validate_citations(hallucinated_response, valid_sources)
    assert "[SYSTEM WARNING" in validated
    assert "99" in validated

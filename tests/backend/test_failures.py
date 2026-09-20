import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app

client = TestClient(app)

def test_empty_retrieval_insufficient_evidence():
    response = client.post("/api/v1/chat", json={
        "query": "What is the exact name of the fictional alien species in this nonexistent document?",
        "mode": "rag"
    })
    assert response.status_code == 200
    data = response.json()
    assert "I cannot find sufficient evidence" in data["answer"] or not data["grounded"]

@patch("vectorstore.qdrant_client.vector_store.search")
def test_vector_store_failure(mock_search):
    mock_search.side_effect = Exception("VECTOR_STORE_FAILURE")
    response = client.post("/api/v1/chat", json={
        "query": "test query",
        "mode": "rag"
    })
    assert response.status_code == 200
    data = response.json()
    assert "Retrieval error:" in data["answer"] or "VECTOR_STORE_FAILURE" in data["answer"]

@patch("providers.llm.llm_provider.generate")
def test_model_failure(mock_generate):
    mock_generate.side_effect = Exception("MODEL_FAILURE")
    response = client.post("/api/v1/chat", json={
        "query": "test query",
        "mode": "direct"
    })
    assert response.status_code == 200
    data = response.json()
    assert "Error generating response" in data["answer"] or "MODEL_FAILURE" in data["answer"]

def test_invalid_citation():
    from verification.grounding import validate_citations
    hallucinated_response = "Here is an answer [Source 99]."
    valid_sources = [{"id": "1", "filename": "test.txt", "text": "content"}]
    validated = validate_citations(hallucinated_response, valid_sources)
    assert "[SYSTEM WARNING" in validated
    assert "99" in validated

import pytest
from retrieval.pipeline import retrieve_documents
from vectorstore.qdrant_client import vector_store
from vectorstore.bm25_store import bm25_store
import time
import uuid

def test_full_retrieval_pipeline():
    # Setup dummy data in Qdrant and BM25 directly to bypass LLM and FastAPI
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a field of artificial intelligence.",
        "Retrieval augmented generation enhances LLM context."
    ]
    metadatas = [
        {"source": "doc1.txt", "page": 1, "type": "document"},
        {"source": "doc2.txt", "page": 1, "type": "document"},
        {"source": "doc3.txt", "page": 1, "type": "document"}
    ]
    
    # Ingest
    doc_ids = vector_store.add_texts(texts, metadatas)
    bm25_store.add_texts(texts, metadatas, doc_ids)
    
    # Test Retrieval
    start = time.time()
    context, sources = retrieve_documents("What enhances LLM context?", limit=2)
    latency = time.time() - start
    
    print(f"Retrieval Latency: {latency:.4f}s")
    
    assert len(sources) > 0
    assert "Retrieval augmented generation" in context
    
    # Cleanup (Optional, since Qdrant is persistent, we might just leave or delete)
    vector_store.client.delete(
        collection_name=vector_store.collection_name,
        points_selector=doc_ids
    )

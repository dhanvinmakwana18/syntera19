import pytest
from core.container import build_container
from core.domain import Query
from retrieval.assembler import PipelineContextAssembler
import time
import uuid

def test_full_retrieval_pipeline():
    container = build_container()
    dense_retriever = container.get_vector_store()
    vector_store = dense_retriever.vector_store
    bm25_store = container.get_bm25_store()
    
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
    import uuid
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    from providers.embeddings import EmbeddingProvider
    emb = EmbeddingProvider()
    vectors = emb.embed_texts(texts)
    payloads = [{"text": t, **m} for t, m in zip(texts, metadatas)]
    vector_store.add_points(vectors=vectors, payloads=payloads, ids=doc_ids)
    
    # For BM25, we use the legacy `add_texts` or update the API
    bm25_store.add_texts(texts, metadatas, doc_ids)
    
    # Test Retrieval
    start = time.time()
    pipeline = container.build_pipeline(retrieval_mode="hybrid")
    result = pipeline.run(Query(text="What enhances LLM context?"), limit=2)
    context_res = PipelineContextAssembler().assemble(result.candidates)
    context = context_res.text
    sources = context_res.sources
    latency = time.time() - start
    
    print(f"Retrieval Latency: {latency:.4f}s")
    
    assert len(sources) > 0
    assert "Retrieval augmented generation" in context
    
    vector_store.client.delete(
        collection_name=vector_store.collection_name,
        points_selector=doc_ids
    )

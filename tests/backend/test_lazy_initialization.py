import pytest
import threading
from core.container import build_container
from intelligence.providers.hf_provider import HuggingFaceProvider
from providers.embeddings import EmbeddingProvider
from retrieval.reranker import CrossEncoderReranker

def test_lazy_initialization_providers():
    # 1. Instantiate the providers
    hf = HuggingFaceProvider()
    emb = EmbeddingProvider()
    reranker = CrossEncoderReranker()
    
    # Internal models should be None until actually fetched
    assert hf._pipeline is None, "HuggingFaceProvider loaded model eagerly"
    assert emb.model is None, "EmbeddingProvider loaded model eagerly"
    assert reranker.model is None, "CrossEncoderReranker loaded model eagerly"

def test_container_build_is_lazy():
    container = build_container()
    
    llm = container.get_intelligence()
    vector_store = container.get_vector_store()
    rr = container.get_reranker()
    
    # Providers returned by the container should also be uninitialized
    assert getattr(llm, '_pipeline', None) is None, "Container eagerly loaded LLM"
    # EmbeddingProvider is inside DenseRetriever -> QdrantVectorStore
    emb = getattr(vector_store, 'embedding_provider', None)
    if emb and isinstance(emb, EmbeddingProvider):
        assert emb.model is None, "Container eagerly loaded EmbeddingProvider"
    if isinstance(rr, CrossEncoderReranker):
        assert rr.model is None, "Container eagerly loaded Reranker"

def test_concurrent_lazy_initialization_safety():
    """
    Proves that concurrent attempts to fetch the model don't result 
    in multiple eager loading calls. 
    We use EmbeddingProvider with a mocked SentenceTransformer to verify.
    """
    provider = EmbeddingProvider()
    
    # Track instantiations
    load_count = 0
    
    def fake_get_model():
        nonlocal load_count
        if provider.model is None:
            with provider._lock:
                if provider.model is None:
                    # Simulate load delay
                    import time
                    time.sleep(0.1)
                    load_count += 1
                    provider.model = "MOCK_MODEL"
        return provider.model
        
    # Patch the internal getter just for tracking
    original_get = provider._get_model
    provider._get_model = fake_get_model
    
    # Run 10 threads trying to fetch the model simultaneously
    threads = []
    for _ in range(10):
        t = threading.Thread(target=provider._get_model)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert load_count == 1, f"Model was instantiated {load_count} times! Lock failed."
    assert provider.model == "MOCK_MODEL"

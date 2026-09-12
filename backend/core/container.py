import copy
from typing import Dict, Any, List
from core.registry import registry
from retrieval.engine import RetrievalPipeline
from retrieval.assembler import ContextBuilder

class ApplicationContainer:
    """Dependency Injection Container for Syntera"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Singleton instances maintained by container
        self._instances = {}
        
    def get_embedding_provider(self, name: str = "sentence_transformers"):
        if "embedding_provider" not in self._instances:
            self._instances["embedding_provider"] = registry.get_embedding_provider(name)
        return self._instances["embedding_provider"]

    def get_vector_store(self, name: str = "qdrant"):
        if "vector_store" not in self._instances:
            emb = self.get_embedding_provider()
            self._instances["vector_store"] = registry.get_retriever(name, embedding_provider=emb)
        return self._instances["vector_store"]
        
    def get_bm25_store(self, name: str = "bm25"):
        if "bm25_store" not in self._instances:
            self._instances["bm25_store"] = registry.get_retriever(name)
        return self._instances["bm25_store"]

    def get_llm(self, name: str = "default"):
        if "llm" not in self._instances:
            self._instances["llm"] = registry.get_llm(name)
        return self._instances["llm"]

    def get_reranker(self, name: str = "cross_encoder"):
        if "reranker" not in self._instances:
            self._instances["reranker"] = registry.get_reranker(name)
        return self._instances["reranker"]

    def build_pipeline(self, retrieval_mode: str = "rerank", expand_neighbors: bool = False, dense_weight: float = 1.0, sparse_weight: float = 1.0) -> RetrievalPipeline:
        """Constructs a transient RetrievalPipeline per request, injected with correct dependencies."""
        retrievers = []
        if retrieval_mode in ["dense", "hybrid", "rerank"]:
            retrievers.append(self.get_vector_store())
        if retrieval_mode in ["sparse", "hybrid", "rerank"]:
            retrievers.append(self.get_bm25_store())

        fusion_strategy = None
        if len(retrievers) > 1:
            fusion_strategy = registry.get_fusion("rrf", weights=[dense_weight, sparse_weight])

        reranker = None
        if retrieval_mode == "rerank":
            reranker = self.get_reranker()

        post_processors = []
        if expand_neighbors:
            # Requires access to raw vector store for DB lookup
            # Since vector_store wraps the actual QdrantStore, we need to pass the raw store
            # In Phase 4, the DenseRetriever has .vector_store
            dense_r = self.get_vector_store()
            pp = registry.get_post_processor("neighbor_expansion", vector_store=dense_r.vector_store)
            post_processors.append(pp)

        return RetrievalPipeline(
            retrievers=retrievers,
            fusion_strategy=fusion_strategy,
            reranker=reranker,
            post_processors=post_processors
        )

# Global bootstrap function
def build_container() -> ApplicationContainer:
    """Bootstraps the application container (called during FastAPI lifespan)."""
    # Import components so they register themselves with the registry
    import vectorstore.qdrant_client
    import vectorstore.bm25_store
    import providers.llm
    import providers.embeddings
    import retrieval.fusion
    import retrieval.post_processors
    import retrieval.reranker
    
    return ApplicationContainer()

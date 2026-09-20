from vectorstore.qdrant_client import vector_store
from vectorstore.bm25_store import bm25_store
from core.config import settings
from retrieval.query_transform import transform_query
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import reranker_service
from retrieval.assembler import assemble_context

def retrieve_documents(
    query: str, 
    limit: int = None, 
    retrieval_mode: str = "rerank", 
    dense_weight: float = None, 
    sparse_weight: float = None, 
    candidate_depth: int = 20, 
    expand_neighbors: bool = False
):
    """
    Retrieves documents. Modes: dense, sparse, hybrid, rerank.
    Uses configurable environment defaults if values are not provided.
    """
    limit = limit if limit is not None else settings.RETRIEVAL_K
    dense_weight = dense_weight if dense_weight is not None else settings.DENSE_WEIGHT
    sparse_weight = sparse_weight if sparse_weight is not None else settings.SPARSE_WEIGHT
    
    optimized_query = transform_query(query)
    if not optimized_query:
        return "", []
        
    if not bm25_store._is_synced:
        bm25_store.sync_from_qdrant(vector_store)

    dense_cands = vector_store.search(optimized_query, limit=candidate_depth) if retrieval_mode in ["dense", "hybrid", "rerank"] else []
    sparse_cands = bm25_store.search(optimized_query, limit=candidate_depth) if retrieval_mode in ["sparse", "hybrid", "rerank"] else []
    
    if retrieval_mode == "dense":
        candidates = dense_cands
        for c in candidates: c["rerank_score"] = c["score"]
    elif retrieval_mode == "sparse":
        candidates = sparse_cands
        for c in candidates: c["rerank_score"] = c["score"]
    else:
        # Hybrid or Rerank
        fused = reciprocal_rank_fusion(dense_cands, sparse_cands, limit=candidate_depth, dense_weight=dense_weight, sparse_weight=sparse_weight)
        if retrieval_mode == "hybrid":
            candidates = fused
            for c in candidates: c["rerank_score"] = c.get("rrf_score", 0)
        else:
            candidates = reranker_service.rerank(optimized_query, fused, limit=limit)
    
    candidates = candidates[:limit]
    context, sources = assemble_context(candidates, relevance_threshold=None, expand_neighbors=expand_neighbors)
    return context, sources

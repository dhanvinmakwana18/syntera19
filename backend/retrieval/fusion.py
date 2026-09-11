def reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, limit=5, dense_weight=1.0, sparse_weight=1.0):
    """
    Fuses dense and sparse candidate lists using Reciprocal Rank Fusion (RRF).
    
    dense_candidates: List of dicts, e.g. [{"id": "...", "score": 0.8, "payload": {...}}, ...]
    sparse_candidates: List of dicts, e.g. [{"id": "...", "score": 2.5, "payload": {...}}, ...]
    k: RRF constant (default 60 is standard)
    limit: Number of final candidates to return
    dense_weight: Weight applied to dense candidates RRF contribution
    sparse_weight: Weight applied to sparse candidates RRF contribution
    """
    rrf_scores = {}
    
    def process_candidates(candidates, weight):
        for rank, candidate in enumerate(candidates):
            doc_id = candidate["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "score": 0.0,
                    "candidate": candidate
                }
            # Weighted RRF formula: weight * (1 / (k + rank + 1))
            rrf_scores[doc_id]["score"] += weight * (1.0 / (k + rank + 1))

    process_candidates(dense_candidates, dense_weight)
    process_candidates(sparse_candidates, sparse_weight)
    
    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Return top N candidates, preserving ID and payload, adding rrf_score
    results = []
    for item in sorted_items[:limit]:
        result = item["candidate"].copy()
        result["rrf_score"] = item["score"]
        # Explicitly tag retrieval method if we want, but since they are fused, we just keep RRF score.
        results.append(result)
        
    return results

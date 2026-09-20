from core.contracts import BaseFusionStrategy
from core.domain import RetrievalResult
from typing import List, Dict

class RRFFusionStrategy(BaseFusionStrategy):
    """Domain-compliant RRF fusion strategy taking N candidate lists."""
    def __init__(self, k: int = 60, weights: List[float] = None):
        self.k = k
        self.weights = weights

    def fuse(self, candidate_lists: List[List[RetrievalResult]], limit: int = 5, **kwargs) -> List[RetrievalResult]:
        rrf_scores: Dict[str, dict] = {}
        
        weights = self.weights if self.weights else [1.0] * len(candidate_lists)
        if len(weights) != len(candidate_lists):
            # Fallback if lengths mismatch
            weights = [1.0] * len(candidate_lists)
            
        for list_idx, candidates in enumerate(candidate_lists):
            weight = weights[list_idx]
            for rank, candidate in enumerate(candidates):
                doc_id = candidate.node.id
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {
                        "score": 0.0,
                        "node": candidate.node
                    }
                rrf_scores[doc_id]["score"] += weight * (1.0 / (self.k + rank + 1))
                
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        
        results = []
        for item in sorted_items[:limit]:
            results.append(RetrievalResult(node=item["node"], score=item["score"]))
            
        return results

def reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, limit=5, dense_weight=1.0, sparse_weight=1.0):
    """
    Legacy implementation: Fuses dense and sparse candidate lists using Reciprocal Rank Fusion (RRF).
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
        results.append(result)
        
    return results
# PHASE 4: Component Registration

def create_rrf_fusion(**kwargs) -> BaseFusionStrategy:
    return RRFFusionStrategy()




def register(registry):
    registry.register_fusion('rrf', create_rrf_fusion)

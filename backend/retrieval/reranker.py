from sentence_transformers import CrossEncoder
from core.contracts import BaseReranker
from core.domain import Query, RetrievalResult
from typing import List
import copy

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self.model = None
        self.load_failed = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"Warning: Failed to load reranker {self.model_name}. Reranking will be disabled. Error: {e}")
            self.model = None
            self.load_failed = True

    def rerank(self, query: Query, candidates: List[RetrievalResult], limit: int = 5) -> List[RetrievalResult]:
        if not candidates:
            return []
            
        if self.model is None:
            return candidates[:limit]
            
        pairs = [[query.text, doc.node.text] for doc in candidates]
        scores = self.model.predict(pairs)
        
        results = []
        for doc, score in zip(candidates, scores):
            # Create a new RetrievalResult with updated score
            results.append(RetrievalResult(node=doc.node, score=float(score)))
            
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

class RerankerService:
    """Legacy RerankerService maintaining old API while using new contracts internally."""
    def __init__(self):
        self._reranker = CrossEncoderReranker()
        
    @property
    def load_failed(self):
        return self._reranker.load_failed

    def rerank(self, query: str, candidates: list, limit: int = 5):
        """Legacy rerank modifying dicts in place."""
        if not candidates:
            return []
            
        if self._reranker.model is None:
            # Fallback for legacy format
            for c in candidates:
                c["rerank_score"] = c.get("rrf_score", c.get("score", 0.0))
            return candidates[:limit]
            
        pairs = [[query, doc["payload"]["text"]] for doc in candidates]
        scores = self._reranker.model.predict(pairs)
        
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:limit]

# -------------------------------------------------------------
# PHASE 4: Component Registration
# -------------------------------------------------------------
from core.registry import registry

def create_cross_encoder_reranker() -> BaseReranker:
    return CrossEncoderReranker()

registry.register_reranker("cross_encoder", create_cross_encoder_reranker)

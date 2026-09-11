from sentence_transformers import CrossEncoder

class RerankerService:
    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
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
            
    def rerank(self, query: str, candidates: list, limit: int = 5):
        """
        Reranks a list of candidates using a Cross-Encoder.
        If the model is unavailable, gracefully falls back to the original order.
        """
        if not candidates:
            return []
            
        if self.model is None:
            # Fallback: Just return candidates as-is, truncated to limit
            # but mark that reranking didn't occur (or use their base score)
            for c in candidates:
                c["rerank_score"] = c.get("rrf_score", c.get("score", 0.0))
            return candidates[:limit]
            
        # Prepare pairs for cross-encoder
        pairs = [[query, doc["payload"]["text"]] for doc in candidates]
        scores = self.model.predict(pairs)
        
        # Merge scores back and sort
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:limit]

reranker_service = RerankerService()

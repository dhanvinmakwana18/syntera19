from typing import List, Optional, Dict, Any
from core.domain import Query, RetrievalResult
from core.contracts import BaseRetriever, BaseFusionStrategy, BaseReranker, BaseNodePostProcessor
import time

class PipelineResult:
    def __init__(self, candidates: List[RetrievalResult], trace: List[Dict[str, Any]]):
        self.candidates = candidates
        self.trace = trace

class RetrievalPipeline:
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        fusion_strategy: Optional[BaseFusionStrategy] = None,
        reranker: Optional[BaseReranker] = None,
        post_processors: Optional[List[BaseNodePostProcessor]] = None
    ):
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker
        self.post_processors = post_processors or []
        
    def run(self, query: Query, limit: int = 5, **kwargs) -> PipelineResult:
        trace = []
        
        # 1. Retrieval
        candidate_lists = []
        for i, retriever in enumerate(self.retrievers):
            start = time.time()
            try:
                candidates = retriever.retrieve(query, limit=limit, **kwargs)
                latency = time.time() - start
                candidate_lists.append(candidates)
                trace.append({
                    "stage": f"retrieval_{i}",
                    "retriever": retriever.__class__.__name__,
                    "latency": latency,
                    "count": len(candidates),
                    "status": "success"
                })
            except Exception as e:
                latency = time.time() - start
                trace.append({
                    "stage": f"retrieval_{i}",
                    "retriever": retriever.__class__.__name__,
                    "latency": latency,
                    "error": str(e),
                    "status": "failure"
                })
                
        # If no lists succeeded, handle gracefully or raise if all failed
        if not candidate_lists:
            errors = [t["error"] for t in trace if t.get("status") == "failure" and "error" in t]
            if errors and len(errors) == len(self.retrievers):
                raise RuntimeError(f"All retrievers failed. Primary error: {errors[0]}")
            return PipelineResult([], trace)
            
        # 2. Fusion
        fused_candidates = []
        if len(candidate_lists) > 1 and self.fusion_strategy:
            start = time.time()
            try:
                fused_candidates = self.fusion_strategy.fuse(candidate_lists, limit=limit, **kwargs)
                latency = time.time() - start
                trace.append({
                    "stage": "fusion",
                    "strategy": self.fusion_strategy.__class__.__name__,
                    "latency": latency,
                    "count": len(fused_candidates),
                    "status": "success"
                })
            except Exception as e:
                latency = time.time() - start
                trace.append({
                    "stage": "fusion",
                    "strategy": self.fusion_strategy.__class__.__name__,
                    "latency": latency,
                    "error": str(e),
                    "status": "failure"
                })
                # Fallback to just the first successful list
                fused_candidates = candidate_lists[0]
        elif len(candidate_lists) == 1:
            fused_candidates = candidate_lists[0]
        else:
            # multiple lists but no fusion strategy configured
            # naive merge and sort
            fused_candidates = [c for lst in candidate_lists for c in lst]
            fused_candidates.sort(key=lambda x: x.score, reverse=True)
            fused_candidates = fused_candidates[:limit]
            
        # 3. Reranking
        if self.reranker and fused_candidates:
            start = time.time()
            try:
                fused_candidates = self.reranker.rerank(query, fused_candidates, limit=limit)
                latency = time.time() - start
                trace.append({
                    "stage": "reranking",
                    "reranker": self.reranker.__class__.__name__,
                    "latency": latency,
                    "count": len(fused_candidates),
                    "status": "success"
                })
            except Exception as e:
                latency = time.time() - start
                trace.append({
                    "stage": "reranking",
                    "reranker": self.reranker.__class__.__name__,
                    "latency": latency,
                    "error": str(e),
                    "status": "failure"
                })
                
        # 4. Post Processing (e.g. Neighbor Expansion)
        for i, processor in enumerate(self.post_processors):
            start = time.time()
            try:
                fused_candidates = processor.process(fused_candidates, **kwargs)
                latency = time.time() - start
                trace.append({
                    "stage": f"post_processing_{i}",
                    "processor": processor.__class__.__name__,
                    "latency": latency,
                    "count": len(fused_candidates),
                    "status": "success"
                })
            except Exception as e:
                latency = time.time() - start
                trace.append({
                    "stage": f"post_processing_{i}",
                    "processor": processor.__class__.__name__,
                    "latency": latency,
                    "error": str(e),
                    "status": "failure"
                })
                
        return PipelineResult(fused_candidates, trace)

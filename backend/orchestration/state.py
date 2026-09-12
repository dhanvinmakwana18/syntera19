from typing import Optional, List, Dict, Any
from pydantic import Field
from core.graph.contracts import GraphState
from core.domain import Query, RetrievalResult, GenerationContext, GenerationResult, VerificationResult

class RAGState(GraphState):
    query: Optional[Query] = None
    processed_queries: List[Query] = []
    retrieval_results: List[RetrievalResult] = []
    context: Optional[GenerationContext] = None
    generation_result: Optional[GenerationResult] = None
    verification_result: Optional[VerificationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

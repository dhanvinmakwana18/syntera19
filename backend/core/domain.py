from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Node(BaseModel):
    """Represents a chunk of text or a document retrieved from a store."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0

class RetrievalResult(BaseModel):
    """Represents the output of a retrieval or reranking step."""
    node: Node
    score: float

class Query(BaseModel):
    """Represents a query against the system."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerationContext(BaseModel):
    """Represents the assembled context for generation."""
    text: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class GenerationResult(BaseModel):
    """Represents the output of the generation step."""
    answer: str
    raw_response: str = ""
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)

class VerificationResult(BaseModel):
    """Represents the output of a verification step."""
    passed: bool
    reason: str
    metrics: Dict[str, Any] = Field(default_factory=dict)

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

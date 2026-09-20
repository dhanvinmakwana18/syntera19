from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from core.domain import Query, RetrievalResult, Node

class BaseEmbeddingProvider(ABC):
    """Contract for generating embeddings."""
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
        
    @property
    @abstractmethod
    def vector_size(self) -> int:
        pass


class BaseVectorStore(ABC):
    """Contract for purely vector-based storage and retrieval operations.
    Must not internally generate embeddings."""
    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        pass
        
    @abstractmethod
    def add_points(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        pass
        
    @abstractmethod
    def get_points_by_ids(self, ids: List[str]) -> List[Node]:
        pass


class BaseRetriever(ABC):
    """Contract for retrieving candidate nodes for a query."""
    @abstractmethod
    def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        pass


class BaseFusionStrategy(ABC):
    """Contract for fusing candidates from multiple retrievers."""
    @abstractmethod
    def fuse(self, candidate_lists: List[List[RetrievalResult]], limit: int = 5, **kwargs) -> List[RetrievalResult]:
        pass


class BaseReranker(ABC):
    """Contract for reranking retrieved candidates."""
    @abstractmethod
    def rerank(self, query: Query, candidates: List[RetrievalResult], limit: int = 5) -> List[RetrievalResult]:
        pass

class BaseNodePostProcessor(ABC):
    """Contract for processing nodes after retrieval/reranking."""
    @abstractmethod
    def process(self, nodes: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        pass

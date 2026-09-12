from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from core.domain import Query, RetrievalResult, Node, GenerationContext, GenerationResult, VerificationResult

class BaseEmbeddingProvider(ABC):
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
    @abstractmethod
    def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        pass


class BaseFusionStrategy(ABC):
    @abstractmethod
    def fuse(self, candidate_lists: List[List[RetrievalResult]], limit: int = 5, **kwargs) -> List[RetrievalResult]:
        pass


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: Query, candidates: List[RetrievalResult], limit: int = 5) -> List[RetrievalResult]:
        pass

class BaseNodePostProcessor(ABC):
    @abstractmethod
    def process(self, nodes: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        pass

class BaseQueryProcessor(ABC):
    @abstractmethod
    def process(self, query: Query, **kwargs) -> List[Query]:
        pass

class BaseContextAssembler(ABC):
    @abstractmethod
    def assemble(self, results: List[RetrievalResult], **kwargs) -> GenerationContext:
        pass

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: Query, context: GenerationContext, **kwargs) -> GenerationResult:
        pass

class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, query: Query, context: GenerationContext, response: GenerationResult, **kwargs) -> VerificationResult:
        pass

class BaseGraphNode(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @abstractmethod
    def execute(self, state: Any) -> Dict[str, Any]:
        pass

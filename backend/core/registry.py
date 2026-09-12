from typing import Callable, Dict, Any, Type
from core.contracts import BaseRetriever, BaseFusionStrategy, BaseReranker, BaseEmbeddingProvider, BaseNodePostProcessor

class ComponentRegistry:
    def __init__(self):
        self._retrievers: Dict[str, Callable[..., BaseRetriever]] = {}
        self._fusions: Dict[str, Callable[..., BaseFusionStrategy]] = {}
        self._rerankers: Dict[str, Callable[..., BaseReranker]] = {}
        self._embedding_providers: Dict[str, Callable[..., BaseEmbeddingProvider]] = {}
        self._post_processors: Dict[str, Callable[..., BaseNodePostProcessor]] = {}
        self._llms: Dict[str, Callable[..., Any]] = {}
        
    def register_retriever(self, name: str, factory: Callable[..., BaseRetriever]):
        self._retrievers[name] = factory
        
    def get_retriever(self, name: str, **kwargs) -> BaseRetriever:
        if name not in self._retrievers:
            raise ValueError(f"Retriever '{name}' not found in registry.")
        return self._retrievers[name](**kwargs)

    def register_fusion(self, name: str, factory: Callable[..., BaseFusionStrategy]):
        self._fusions[name] = factory
        
    def get_fusion(self, name: str, **kwargs) -> BaseFusionStrategy:
        if name not in self._fusions:
            raise ValueError(f"Fusion strategy '{name}' not found in registry.")
        return self._fusions[name](**kwargs)

    def register_reranker(self, name: str, factory: Callable[..., BaseReranker]):
        self._rerankers[name] = factory
        
    def get_reranker(self, name: str, **kwargs) -> BaseReranker:
        if name not in self._rerankers:
            raise ValueError(f"Reranker '{name}' not found in registry.")
        return self._rerankers[name](**kwargs)
        
    def register_embedding_provider(self, name: str, factory: Callable[..., BaseEmbeddingProvider]):
        self._embedding_providers[name] = factory
        
    def get_embedding_provider(self, name: str, **kwargs) -> BaseEmbeddingProvider:
        if name not in self._embedding_providers:
            raise ValueError(f"Embedding Provider '{name}' not found in registry.")
        return self._embedding_providers[name](**kwargs)

    def register_post_processor(self, name: str, factory: Callable[..., BaseNodePostProcessor]):
        self._post_processors[name] = factory

    def get_post_processor(self, name: str, **kwargs) -> BaseNodePostProcessor:
        if name not in self._post_processors:
            raise ValueError(f"Post Processor '{name}' not found in registry.")
        return self._post_processors[name](**kwargs)
        
    def register_llm(self, name: str, factory: Callable[..., Any]):
        self._llms[name] = factory
        
    def get_llm(self, name: str, **kwargs) -> Any:
        if name not in self._llms:
            raise ValueError(f"LLM '{name}' not found in registry.")
        return self._llms[name](**kwargs)

# Global registry instance strictly for registration (not state)
registry = ComponentRegistry()

from typing import Callable, Dict, Any, Type
from core.contracts import (
    BaseRetriever, BaseFusionStrategy, BaseReranker, 
    BaseEmbeddingProvider, BaseNodePostProcessor,
    BaseQueryProcessor, BaseContextAssembler, 
    BaseGenerator, BaseVerifier
)

class ComponentRegistry:
    def __init__(self):
        self._retrievers: Dict[str, Callable[..., BaseRetriever]] = {}
        self._fusions: Dict[str, Callable[..., BaseFusionStrategy]] = {}
        self._rerankers: Dict[str, Callable[..., BaseReranker]] = {}
        self._embedding_providers: Dict[str, Callable[..., BaseEmbeddingProvider]] = {}
        self._post_processors: Dict[str, Callable[..., BaseNodePostProcessor]] = {}
        self._llms: Dict[str, Callable[..., Any]] = {}
        
        self._query_processors: Dict[str, Callable[..., BaseQueryProcessor]] = {}
        self._context_assemblers: Dict[str, Callable[..., BaseContextAssembler]] = {}
        self._generators: Dict[str, Callable[..., BaseGenerator]] = {}
        self._verifiers: Dict[str, Callable[..., BaseVerifier]] = {}
        
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

    # New Registries
    def register_query_processor(self, name: str, factory: Callable[..., BaseQueryProcessor]):
        self._query_processors[name] = factory
        
    def get_query_processor(self, name: str, **kwargs) -> BaseQueryProcessor:
        if name not in self._query_processors:
            raise ValueError(f"Query Processor '{name}' not found in registry.")
        return self._query_processors[name](**kwargs)
        
    def register_context_assembler(self, name: str, factory: Callable[..., BaseContextAssembler]):
        self._context_assemblers[name] = factory
        
    def get_context_assembler(self, name: str, **kwargs) -> BaseContextAssembler:
        if name not in self._context_assemblers:
            raise ValueError(f"Context Assembler '{name}' not found in registry.")
        return self._context_assemblers[name](**kwargs)
        
    def register_generator(self, name: str, factory: Callable[..., BaseGenerator]):
        self._generators[name] = factory
        
    def get_generator(self, name: str, **kwargs) -> BaseGenerator:
        if name not in self._generators:
            raise ValueError(f"Generator '{name}' not found in registry.")
        return self._generators[name](**kwargs)
        
    def register_verifier(self, name: str, factory: Callable[..., BaseVerifier]):
        self._verifiers[name] = factory
        
    def get_verifier(self, name: str, **kwargs) -> BaseVerifier:
        if name not in self._verifiers:
            raise ValueError(f"Verifier '{name}' not found in registry.")
        return self._verifiers[name](**kwargs)


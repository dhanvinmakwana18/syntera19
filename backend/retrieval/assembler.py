from typing import List, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod
from core.domain import RetrievalResult, GenerationContext
from core.contracts import BaseContextAssembler

class ContextFilter(ABC):
    @abstractmethod
    def filter(self, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        pass

class ContextSorter(ABC):
    @abstractmethod
    def sort(self, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        pass
        
class ContextFormatter(ABC):
    @abstractmethod
    def format(self, results: List[RetrievalResult], **kwargs) -> GenerationContext:
        pass

class ThresholdFilter(ContextFilter):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        
    def filter(self, results: List[RetrievalResult], relevance_threshold: float = None, **kwargs) -> List[RetrievalResult]:
        t = relevance_threshold if relevance_threshold is not None else self.threshold
        return [r for r in results if r.score >= t]

class GroupAndRankSorter(ContextSorter):
    def sort(self, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        grouped: Dict[str, List[RetrievalResult]] = {}
        for r in results:
            source = r.node.metadata.get("source", "Unknown")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(r)
            
        for source in grouped:
            grouped[source] = sorted(grouped[source], key=lambda x: x.node.metadata.get("chunk_index", 999999))
            
        def max_score(group: List[RetrievalResult]) -> float:
            return max(item.score for item in group)
            
        sorted_sources = sorted(grouped.keys(), key=lambda s: max_score(grouped[s]), reverse=True)
        
        flat_sorted = []
        for s in sorted_sources:
            flat_sorted.extend(grouped[s])
            
        return flat_sorted

class StandardFormatter(ContextFormatter):
    def format(self, results: List[RetrievalResult], **kwargs) -> GenerationContext:
        context = ""
        sources = []
        
        for r in results:
            node = r.node
            text = node.text
            page = node.metadata.get("page", "?")
            section_path = node.metadata.get("section_path", "Root")
            chunk_index = node.metadata.get("chunk_index", "?")
            source_name = node.metadata.get("source", "Unknown")
            
            source_idx = len(sources) + 1
            
            sources.append({
                "id": source_idx,
                "filename": source_name,
                "page": page,
                "section": section_path,
                "chunk_index": chunk_index,
                "text": text,
                "score": r.score
            })
            
            context += f"[Source {source_idx}] (File: {source_name}, Section: {section_path}, Page: {page}):\n{text}\n\n"
            
        return GenerationContext(text=context.strip(), sources=sources)

class PipelineContextAssembler(BaseContextAssembler):
    def __init__(self, filters: List[ContextFilter] = None, sorter: ContextSorter = None, formatter: ContextFormatter = None):
        self.filters = filters or [ThresholdFilter()]
        self.sorter = sorter or GroupAndRankSorter()
        self.formatter = formatter or StandardFormatter()
        
    def assemble(self, results: List[RetrievalResult], **kwargs) -> GenerationContext:
        for f in self.filters:
            results = f.filter(results, **kwargs)
            
        results = self.sorter.sort(results, **kwargs)
        
        return self.formatter.format(results, **kwargs)

def register(registry):
    registry.register_context_assembler("default", lambda **kwargs: PipelineContextAssembler())

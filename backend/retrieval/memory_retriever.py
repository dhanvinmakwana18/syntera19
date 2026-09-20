from typing import List, Optional
from core.contracts import BaseRetriever
from core.domain import Query, RetrievalResult, Node
from memory.contracts import BaseMemoryStore, MemoryType

class MemoryRetriever(BaseRetriever):
    def __init__(self, store: BaseMemoryStore, default_memory_type: Optional[MemoryType] = None):
        self.store = store
        self.default_memory_type = default_memory_type
        
    def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        mem_type = kwargs.get('memory_type', self.default_memory_type)
        records = self.store.search(query.text, memory_type=mem_type, limit=limit)
        
        results = []
        for record in records:
            node = Node(
                id=record.id,
                text=record.content,
                metadata={
                    "type": record.type.value,
                    "source": record.source,
                    "timestamp": record.timestamp.isoformat(),
                    **record.metadata
                },
                score=record.relevance
            )
            results.append(RetrievalResult(node=node, score=record.relevance))
        return results

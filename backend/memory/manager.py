from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from memory.contracts import BaseMemoryManager, BaseMemoryStore, MemoryRecord, MemoryType

class MemoryManager(BaseMemoryManager):
    def __init__(self, store: BaseMemoryStore):
        self._store = store
        
    def observe(self, content: str, source: str = "", metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            type=MemoryType.WORKING,
            content=content,
            timestamp=datetime.utcnow(),
            source=source,
            metadata=metadata or {},
            relevance=1.0
        )
        return record
        
    def validate(self, record: MemoryRecord) -> bool:
        # Simple validation: content must not be empty
        return bool(record.content and record.content.strip())
        
    def store(self, record: MemoryRecord) -> None:
        if not self.validate(record):
            return
            
        # Deduplication in working memory based on exact content match
        existing = self._store.search(record.content, memory_type=record.type, limit=1)
        if existing and existing[0].content == record.content:
            return
            
        self._store.add(record)
        
    def retrieve(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        return self._store.search(query, limit=limit)
        
    def consolidate(self) -> None:
        # Promote WorkingMemory -> Episodic/Semantic based on relevance
        working_memories = self._store.get_all(MemoryType.WORKING)
        for mem in working_memories:
            # Promote if it has high relevance
            if mem.relevance >= 0.5:
                mem.type = MemoryType.EPISODIC
                self._store.update(mem)
                
    def forget(self) -> None:
        # Delete expired memories
        now = datetime.utcnow()
        all_mems = self._store.get_all()
        for mem in all_mems:
            if mem.expiration and mem.expiration < now:
                self._store.delete(mem.id)
            elif mem.type == MemoryType.WORKING and mem.relevance < 0.1:
                self._store.delete(mem.id)

from typing import List, Optional, Dict
import uuid
from memory.contracts import BaseMemoryStore, MemoryRecord, MemoryType

class InMemoryDictStore(BaseMemoryStore):
    def __init__(self):
        self._records: Dict[str, MemoryRecord] = {}
        
    def add(self, record: MemoryRecord) -> str:
        if not record.id:
            record.id = str(uuid.uuid4())
        self._records[record.id] = record
        return record.id
        
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self._records.get(record_id)
        
    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 5) -> List[MemoryRecord]:
        # Simple string matching for search
        results = []
        for r in self._records.values():
            if memory_type and r.type != memory_type:
                continue
            if query.lower() in r.content.lower():
                results.append(r)
        
        # Sort by timestamp (descending) as a heuristic
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]
        
    def update(self, record: MemoryRecord) -> None:
        if record.id in self._records:
            self._records[record.id] = record
            
    def delete(self, record_id: str) -> None:
        if record_id in self._records:
            del self._records[record_id]
            
    def get_all(self, memory_type: Optional[MemoryType] = None) -> List[MemoryRecord]:
        if memory_type:
            return [r for r in self._records.values() if r.type == memory_type]
        return list(self._records.values())

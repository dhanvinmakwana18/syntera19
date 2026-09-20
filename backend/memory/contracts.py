from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class MemoryType(str, Enum):
    WORKING = "WorkingMemory"
    EPISODIC = "EpisodicMemory"
    SEMANTIC = "SemanticMemory"
    LONG_TERM = "LongTermMemory"

class MemoryRecord(BaseModel):
    id: str
    type: MemoryType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = ""
    provenance: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance: float = 1.0
    expiration: Optional[datetime] = None

class BaseMemoryStore(ABC):
    @abstractmethod
    def add(self, record: MemoryRecord) -> str:
        pass
        
    @abstractmethod
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        pass
        
    @abstractmethod
    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 5) -> List[MemoryRecord]:
        pass
        
    @abstractmethod
    def update(self, record: MemoryRecord) -> None:
        pass
        
    @abstractmethod
    def delete(self, record_id: str) -> None:
        pass
        
    @abstractmethod
    def get_all(self, memory_type: Optional[MemoryType] = None) -> List[MemoryRecord]:
        pass

class BaseMemoryManager(ABC):
    @abstractmethod
    def observe(self, content: str, source: str = "", metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """Observation phase."""
        pass
        
    @abstractmethod
    def validate(self, record: MemoryRecord) -> bool:
        """Validation phase."""
        pass
        
    @abstractmethod
    def store(self, record: MemoryRecord) -> None:
        """Storage phase."""
        pass
        
    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        """Retrieval phase."""
        pass
        
    @abstractmethod
    def consolidate(self) -> None:
        """Update/Consolidation phase."""
        pass
        
    @abstractmethod
    def forget(self) -> None:
        """Forgetting phase."""
        pass

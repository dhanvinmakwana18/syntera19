import uuid
import time
import threading
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.graph.contracts import GraphState
from core.domain import Query
from core.creation.domain import ModelRequirement, ToolRequirement

class SwarmMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    sender: str
    recipient: Optional[str] = None
    content: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MessageLedger(BaseModel):
    messages: List[SwarmMessage] = Field(default_factory=list)
    _lock: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self._lock = threading.Lock()

    def add(self, message: SwarmMessage):
        with self._lock:
            self.messages.append(message)
            
    def get_all(self) -> List[SwarmMessage]:
        with self._lock:
            return list(self.messages)

    def get_by_sender(self, sender: str) -> List[SwarmMessage]:
        with self._lock:
            return [m for m in self.messages if m.sender == sender]

    def get_by_recipient(self, recipient: str) -> List[SwarmMessage]:
        with self._lock:
            return [m for m in self.messages if m.recipient == recipient]

class SwarmState(GraphState):
    query: Query
    ledger: MessageLedger = Field(default_factory=MessageLedger)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    events: List[Any] = Field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    # For backward compatibility during migration
    @property
    def messages(self) -> List[SwarmMessage]:
        return self.ledger.get_all()
    
class AgentSpec(BaseModel):
    name: str
    role: str
    system_prompt: str
    model: ModelRequirement
    tools: List[ToolRequirement] = Field(default_factory=list)

class SwarmSpec(BaseModel):
    agents: List[AgentSpec]
    workflow_type: str = "sequential" # sequential, parallel, or custom_graph
    edges: List[List[str]] = Field(default_factory=list) # List of [from_node, to_node]
    entry_point: str
    failure_policy: Optional[str] = None

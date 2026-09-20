from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import uuid

class SwarmEventName(str, Enum):
    START = "SWARM.START"
    AGENT_STARTED = "SWARM.AGENT_STARTED"
    AGENT_COMPLETED = "SWARM.AGENT_COMPLETED"
    AGENT_FAILED = "SWARM.AGENT_FAILED"
    MESSAGE_SENT = "SWARM.MESSAGE_SENT"
    BARRIER_WAIT = "SWARM.BARRIER_WAIT"
    AGENT_SKIPPED = "SWARM.AGENT_SKIPPED"
    COMPLETED = "SWARM.COMPLETED"
    FAILED = "SWARM.FAILED"

class SwarmEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    event_name: SwarmEventName
    swarm_id: Optional[str] = None
    agent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class SwarmObserver:
    def __init__(self):
        self.events: list[SwarmEvent] = []
        
    def emit(self, event_name: SwarmEventName, **kwargs):
        event = SwarmEvent(event_name=event_name, **kwargs)
        self.events.append(event)

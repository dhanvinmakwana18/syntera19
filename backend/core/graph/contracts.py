from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class GraphState(BaseModel):
    """Base interface for graph state. Must be a Pydantic model for validation."""
    pass

class RoutingDecision(BaseModel):
    """Defines which node(s) should be executed next."""
    next_nodes: List[str] = Field(default_factory=list)

class NodeResult(BaseModel):
    """Result of a node execution. Contains state updates and optional routing hints."""
    state_updates: Dict[str, Any] = Field(default_factory=dict)
    routing_decision: Optional[RoutingDecision] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

class GraphNode(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @abstractmethod
    def execute(self, state: Any) -> NodeResult:
        """Executes the node given the current state and returns state updates and routing decisions."""
        pass

class GraphExecutionResult(BaseModel):
    """Final result of a graph execution."""
    final_state: Any
    trace: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None

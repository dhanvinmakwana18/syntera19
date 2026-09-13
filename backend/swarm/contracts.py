from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.graph.contracts import GraphState
from core.domain import Query
from core.creation.domain import ModelRequirement, ToolRequirement

class SwarmMessage(BaseModel):
    sender: str
    recipient: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SwarmState(GraphState):
    query: Query
    messages: List[SwarmMessage] = Field(default_factory=list)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    error: Optional[str] = None
    
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

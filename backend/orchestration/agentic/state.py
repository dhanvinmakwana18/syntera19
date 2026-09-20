from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.graph.contracts import GraphState
from core.domain import Query

class AgentTask(BaseModel):
    id: str
    description: str
    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    result: Optional[str] = None
    
class AgentPlan(BaseModel):
    tasks: List[AgentTask] = Field(default_factory=list)

class AgentState(GraphState):
    query: Query
    plan: Optional[AgentPlan] = None
    current_task_idx: int = 0
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    final_answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3
    error: Optional[str] = None

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SystemIdentity(BaseModel):
    id: str
    name: str
    version: str = "0.1.0"

class CapabilityRequirement(BaseModel):
    name: str
    version: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)

class ModelRequirement(BaseModel):
    task: str
    structured_output: bool = False
    context_window: str = "default"
    local_preferred: bool = False
    reasoning_required: str = "standard"

class ToolRequirement(BaseModel):
    name: str
    version: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeRequirement(BaseModel):
    mode: str = "none" # none, sparse, dense, hybrid
    reranking: bool = False
    expand_neighbors: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)

class EvaluationRequirement(BaseModel):
    metrics: List[str] = Field(default_factory=list)
    criteria: Dict[str, Any] = Field(default_factory=dict)

class AISystemSpecification(BaseModel):
    identity: SystemIdentity
    purpose: str
    capabilities: List[CapabilityRequirement] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    models: List[ModelRequirement] = Field(default_factory=list)
    tools: List[ToolRequirement] = Field(default_factory=list)
    knowledge: Optional[KnowledgeRequirement] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    workflow: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Optional[EvaluationRequirement] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)

class EvaluationPlan(BaseModel):
    system_id: str
    system_version: str
    metrics: List[str]
    criteria: Dict[str, Any]

class GeneratedAISystem(BaseModel):
    specification: AISystemSpecification
    execution_graph: Any  # Cannot cleanly type ExecutionGraph here due to circular imports or Pydantic V2 limitations with arbitrary types, will use Any or configure arbitrary_types_allowed
    evaluation_plan: Optional[EvaluationPlan] = None
    resolved_capabilities: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

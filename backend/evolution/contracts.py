"""
Syntera Evolution Engine - Contracts
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.creation.domain import AISystemSpecification, GeneratedAISystem
from intelligence.evaluation import EvalRunResult

class EvolutionEvent(Enum):
    START = "EVOLUTION.START"
    DIAGNOSIS = "EVOLUTION.DIAGNOSIS"
    CANDIDATE_GENERATED = "EVOLUTION.CANDIDATE_GENERATED"
    CANDIDATE_BUILT = "EVOLUTION.CANDIDATE_BUILT"
    CANDIDATE_EVALUATED = "EVOLUTION.CANDIDATE_EVALUATED"
    COMPARISON = "EVOLUTION.COMPARISON"
    PROMOTED = "EVOLUTION.PROMOTED"
    REJECTED = "EVOLUTION.REJECTED"
    ERROR = "EVOLUTION.ERROR"

class EvolutionDiagnosis(BaseModel):
    """Output of the DiagnosticEngine pinpointing weaknesses."""
    weaknesses: List[str] = Field(description="List of identified weaknesses based on evaluation metrics.")
    suggested_improvements: List[str] = Field(description="List of actionable improvements for the system specification.")
    
class CandidateSpec(BaseModel):
    """A generated variation of an AISystemSpecification."""
    rationale: str = Field(description="Why this specific change should improve the system based on the diagnosis.")
    proposed_spec: AISystemSpecification = Field(description="The modified system specification.")
    
class CandidateStatus(Enum):
    PENDING = "PENDING"
    BUILT = "BUILT"
    EVALUATED = "EVALUATED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

class EvolutionCandidate(BaseModel):
    """Tracks a candidate throughout the evolution lifecycle."""
    id: str
    parent_system_id: str
    proposed_spec: AISystemSpecification
    rationale: str
    status: CandidateStatus = CandidateStatus.PENDING
    built_system: Optional[GeneratedAISystem] = None
    eval_result: Optional[EvalRunResult] = None
    error_message: Optional[str] = None

class SelectionObjective(Enum):
    MAXIMIZE_SUCCESS = "MAXIMIZE_SUCCESS"
    MINIMIZE_LATENCY = "MINIMIZE_LATENCY"
    MINIMIZE_COST = "MINIMIZE_COST"

class SelectionRule(BaseModel):
    objective: SelectionObjective
    weight: float = 1.0
    threshold: Optional[float] = None  # e.g., must be > 0.8 success rate

class SelectionPolicy(BaseModel):
    """Multi-objective policy for ranking evaluation results."""
    rules: List[SelectionRule]
    strict_regression_rejection: bool = True  # Reject if primary metric degrades at all

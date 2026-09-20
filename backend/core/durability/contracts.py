from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import datetime

class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class FailureRecord(BaseModel):
    node: str
    error: str
    attempt: int
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class RetryPolicy(BaseModel):
    max_retries: int = 3
    backoff_factor: float = 1.0
    base_delay: float = 0.5  # seconds

class NodeExecutionRecord(BaseModel):
    node: str
    status: str
    attempts: int
    start_time: float
    end_time: Optional[float] = None
    result_updates: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class Checkpoint(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    state_data: Dict[str, Any]
    completed_nodes: List[str]
    failed_nodes: List[str]
    active_nodes: List[str]
    trace_so_far: List[Dict[str, Any]]
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class ResumePoint(BaseModel):
    workflow_id: str
    active_nodes: List[str]
    state_data: Dict[str, Any]
    trace: List[Dict[str, Any]]

class WorkflowRun(BaseModel):
    id: str
    status: WorkflowStatus
    checkpoints: List[Checkpoint] = Field(default_factory=list)
    failures: List[FailureRecord] = Field(default_factory=list)

class DurableStore(ABC):
    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Saves a checkpoint durably."""
        pass

    @abstractmethod
    def get_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        """Retrieves the latest checkpoint for a workflow run."""
        pass

    @abstractmethod
    def log_failure(self, workflow_id: str, failure: FailureRecord) -> None:
        """Logs a failure event."""
        pass

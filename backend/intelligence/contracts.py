"""
Syntera Intelligence Core — Contracts

Typed request/response contracts for all intelligence operations.
Provider-agnostic. No SDK references.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class IntelligenceRequest(BaseModel):
    """A typed request to the intelligence layer."""
    prompt: str
    system_prompt: str = ""
    task: str = "generate"  # generate, reason, plan, understand, structured_generate
    complexity: TaskComplexity = TaskComplexity.MODERATE
    temperature: float = 0.2
    max_tokens: int = 2048
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntelligenceResponse(BaseModel):
    """A typed response from the intelligence layer."""
    content: str
    model: str = ""
    provider: str = ""
    usage: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class StructuredGenerationRequest(BaseModel):
    """Request for structured (schema-validated) output from a model."""
    prompt: str
    system_prompt: str = ""
    output_schema: Any = None  # Pydantic model class reference
    temperature: float = 0.1
    max_retries: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructuredGenerationResult(BaseModel):
    """Result of structured generation with validation metadata."""
    data: Any = None  # The parsed Pydantic object
    raw_response: str = ""
    model: str = ""
    provider: str = ""
    attempts: int = 1
    validation_errors: List[str] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class ModelCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    VISION = "vision"


class ModelProfile(BaseModel):
    """Describes a model's capabilities and metadata for routing."""
    name: str
    provider: str
    capabilities: List[ModelCapability] = Field(default_factory=list)
    context_window: int = 4096
    cost_tier: str = "low"  # low, medium, high, frontier
    local: bool = False
    default_for: List[str] = Field(default_factory=list)


class BaseIntelligenceProvider(ABC):
    """Abstract base for all intelligence providers."""

    @abstractmethod
    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        pass

    @abstractmethod
    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        pass

    @property
    @abstractmethod
    def profile(self) -> ModelProfile:
        pass

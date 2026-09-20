"""
Syntera Intelligence — Specification Generator

Translates natural language requirements into validated AISystemSpecifications
using structured output generation with Pydantic validation and retry/repair.
"""
import logging
from typing import Optional, List

from intelligence.contracts import StructuredGenerationRequest, StructuredGenerationResult
from intelligence.router import ModelRouter
from core.creation.domain import (
    AISystemSpecification,
    SystemIdentity,
    CapabilityRequirement,
    KnowledgeRequirement,
    EvaluationRequirement,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ParsedSpecification(BaseModel):
    """
    Intermediate schema the LLM must fill.
    Simpler than AISystemSpecification — avoids deep nesting that models struggle with.
    """
    name: str = Field(description="Short name for the AI system")
    purpose: str = Field(description="One-sentence description of what the system does")
    capabilities: List[str] = Field(
        description="List of capability names. Valid options: chat, retrieval, verification, planning, reasoning, tool_use, summarization, coding"
    )
    knowledge_mode: str = Field(
        default="none",
        description="Knowledge retrieval mode: none, sparse, dense, hybrid"
    )
    reranking: bool = Field(default=False, description="Whether to use neural reranking")
    evaluation_metrics: List[str] = Field(
        default_factory=list,
        description="Evaluation metrics: factuality, citation_correctness, retrieval_recall, latency, task_completion, response_quality"
    )


SPEC_SYSTEM_PROMPT = """You are Syntera's AI System Architect.

Given a natural language description of an AI system requirement, you must produce
a structured specification defining the system's name, purpose, capabilities,
knowledge retrieval mode, and evaluation metrics.

Available capabilities (choose from these ONLY):
- chat: conversational text generation
- retrieval: knowledge retrieval from document stores
- verification: citation and factuality verification
- planning: multi-step task planning
- reasoning: chain-of-thought reasoning
- tool_use: external tool execution
- summarization: document summarization
- coding: code generation and analysis

Knowledge modes:
- none: no retrieval needed
- sparse: keyword-based retrieval (BM25)
- dense: semantic vector retrieval
- hybrid: combined sparse + dense retrieval

Evaluation metrics:
- factuality, citation_correctness, retrieval_recall, latency, task_completion, response_quality

Think carefully about what the user actually needs."""


def generate_specification(
    user_request: str,
    router: ModelRouter,
    system_id: str = "auto",
) -> tuple[AISystemSpecification, StructuredGenerationResult]:
    """
    Translate a natural language requirement into a validated AISystemSpecification.

    Returns:
        (specification, generation_result) tuple.
        If generation fails, specification will be None and result.success will be False.
    """
    gen_request = StructuredGenerationRequest(
        prompt=f"User requirement:\n{user_request}",
        system_prompt=SPEC_SYSTEM_PROMPT,
        output_schema=ParsedSpecification,
        temperature=0.1,
        max_retries=3,
    )

    result = router.structured_generate(gen_request)

    if not result.success or result.data is None:
        return None, result

    parsed: ParsedSpecification = result.data

    # Transform the intermediate parsed spec into the full domain specification
    capabilities = [CapabilityRequirement(name=c.strip().lower()) for c in parsed.capabilities]

    knowledge = None
    if parsed.knowledge_mode != "none":
        knowledge = KnowledgeRequirement(
            mode=parsed.knowledge_mode,
            reranking=parsed.reranking,
        )

    evaluation = None
    if parsed.evaluation_metrics:
        evaluation = EvaluationRequirement(metrics=parsed.evaluation_metrics)

    spec = AISystemSpecification(
        identity=SystemIdentity(
            id=system_id if system_id != "auto" else parsed.name.lower().replace(" ", "_"),
            name=parsed.name,
            version="0.1.0",
        ),
        purpose=parsed.purpose,
        capabilities=capabilities,
        knowledge=knowledge,
        evaluation=evaluation,
    )

    return spec, result

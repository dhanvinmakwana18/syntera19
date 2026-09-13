"""
Syntera Intelligence — Core Intelligence Service

High-level intelligence operations that compose the provider abstraction
with the model router for clean domain-level AI operations.
"""
import time
from typing import Any, Optional, Type
from pydantic import BaseModel

from intelligence.contracts import (
    IntelligenceRequest,
    IntelligenceResponse,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    TaskComplexity,
)
from intelligence.router import ModelRouter


class IntelligenceCore:
    """
    High-level intelligence service for Syntera.

    Provides clean task-oriented methods (understand, generate, reason, plan)
    that route through the ModelRouter to the best available provider.
    """

    def __init__(self, router: ModelRouter):
        self.router = router

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        **kwargs,
    ) -> IntelligenceResponse:
        """General text generation."""
        request = IntelligenceRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            task="generate",
            complexity=complexity,
            **kwargs,
        )
        return self.router.generate(request)

    def understand(self, text: str, instruction: str = "") -> IntelligenceResponse:
        """Understand/analyze text input."""
        system = instruction or "Analyze the following text. Extract the key intent, entities, and requirements."
        return self.generate(
            prompt=text,
            system_prompt=system,
            complexity=TaskComplexity.MODERATE,
        )

    def reason(self, prompt: str, context: str = "") -> IntelligenceResponse:
        """Chain-of-thought reasoning."""
        system = (
            "You are a careful reasoning engine. Think step by step. "
            "Show your reasoning process clearly before giving a final answer."
        )
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        return self.generate(
            prompt=full_prompt,
            system_prompt=system,
            complexity=TaskComplexity.COMPLEX,
        )

    def plan(self, goal: str, context: str = "") -> IntelligenceResponse:
        """Generate a structured plan for achieving a goal."""
        system = (
            "You are a planning engine. Given a goal, produce a clear step-by-step plan. "
            "Each step should have: a number, a description, and what tools or capabilities are needed."
        )
        full_prompt = f"Goal: {goal}"
        if context:
            full_prompt = f"Context: {context}\n\n{full_prompt}"
        return self.generate(
            prompt=full_prompt,
            system_prompt=system,
            complexity=TaskComplexity.COMPLEX,
        )

    def structured_generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: str = "",
        max_retries: int = 3,
    ) -> StructuredGenerationResult:
        """Generate structured output validated against a Pydantic schema."""
        request = StructuredGenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=schema,
            max_retries=max_retries,
        )
        return self.router.structured_generate(request)

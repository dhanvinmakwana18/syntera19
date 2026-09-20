"""
Syntera Intelligence — Ollama Provider

Connects to a local Ollama instance for text generation and structured output.
Supports JSON mode for structured generation with Pydantic validation + retry.
"""
import json
import time
import logging
from typing import Any, Optional

import requests
from pydantic import BaseModel, ValidationError

from intelligence.contracts import (
    BaseIntelligenceProvider,
    IntelligenceRequest,
    IntelligenceResponse,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    ModelProfile,
    ModelCapability,
)

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseIntelligenceProvider):
    """Intelligence provider backed by a local Ollama instance."""

    def __init__(self, model: str = "qwen3", base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._profile = ModelProfile(
            name=model,
            provider="ollama",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.REASONING,
            ],
            context_window=32768,
            cost_tier="low",
            local=True,
            default_for=["generate", "reason", "plan", "understand", "structured_generate"],
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        """Generate text via Ollama chat completions endpoint."""
        start = time.time()
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "")
            usage = {}
            if "eval_count" in data:
                usage["output_tokens"] = data["eval_count"]
            if "prompt_eval_count" in data:
                usage["input_tokens"] = data["prompt_eval_count"]
            if "eval_duration" in data:
                usage["tokens_per_sec"] = round(
                    data["eval_count"] / (data["eval_duration"] / 1e9), 2
                ) if data.get("eval_count") and data.get("eval_duration") else 0

            latency = (time.time() - start) * 1000

            return IntelligenceResponse(
                content=content,
                model=self.model,
                provider="ollama",
                usage=usage,
                latency_ms=latency,
                success=True,
            )

        except requests.RequestException as e:
            latency = (time.time() - start) * 1000
            return IntelligenceResponse(
                content="",
                model=self.model,
                provider="ollama",
                latency_ms=latency,
                success=False,
                error=f"Ollama request failed: {e}",
            )

    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        """
        Generate structured output validated against a Pydantic schema.
        Uses Ollama's JSON mode + schema enforcement + retry with repair.
        """
        start = time.time()
        schema_class = request.output_schema
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

        system_prompt = (
            f"{request.system_prompt}\n\n"
            f"You MUST respond with valid JSON that conforms to this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Output ONLY the JSON object. No markdown, no explanation, no extra text."
        )

        validation_errors = []

        for attempt in range(1, request.max_retries + 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt},
            ]

            # If we have validation errors from a previous attempt, add repair context
            if validation_errors:
                repair_msg = (
                    f"Your previous response had validation errors:\n"
                    f"{validation_errors[-1]}\n\n"
                    f"Fix these errors and output valid JSON conforming to the schema."
                )
                messages.append({"role": "user", "content": repair_msg})

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens if hasattr(request, "max_tokens") else 4096,
                },
            }

            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_content = data.get("message", {}).get("content", "")

                # Parse JSON from response
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError as je:
                    # Try to extract JSON from response
                    start_idx = raw_content.find("{")
                    end_idx = raw_content.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        try:
                            parsed = json.loads(raw_content[start_idx:end_idx])
                        except json.JSONDecodeError:
                            validation_errors.append(f"Attempt {attempt}: Invalid JSON: {str(je)}")
                            continue
                    else:
                        validation_errors.append(f"Attempt {attempt}: No JSON found in response")
                        continue

                # Validate against Pydantic schema
                try:
                    validated = schema_class.model_validate(parsed)
                    latency = (time.time() - start) * 1000
                    return StructuredGenerationResult(
                        data=validated,
                        raw_response=raw_content,
                        model=self.model,
                        provider="ollama",
                        attempts=attempt,
                        validation_errors=validation_errors,
                        success=True,
                        latency_ms=latency,
                    )
                except ValidationError as ve:
                    validation_errors.append(f"Attempt {attempt}: Schema validation failed: {str(ve)}")
                    continue

            except requests.RequestException as e:
                validation_errors.append(f"Attempt {attempt}: Request failed: {str(e)}")
                continue

        # All retries exhausted
        latency = (time.time() - start) * 1000
        return StructuredGenerationResult(
            raw_response="",
            model=self.model,
            provider="ollama",
            attempts=request.max_retries,
            validation_errors=validation_errors,
            success=False,
            error=f"Structured generation failed after {request.max_retries} attempts",
            latency_ms=latency,
        )

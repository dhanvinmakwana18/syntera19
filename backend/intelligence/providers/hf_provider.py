"""
Syntera Intelligence — HuggingFace Local Provider

Provides a CPU/GPU local fallback using HuggingFace Transformers.
Requires `transformers` and `torch`.
"""
import time
import json
import logging
from typing import Any, Optional
from pydantic import ValidationError

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


class HuggingFaceProvider(BaseIntelligenceProvider):
    """Intelligence provider backed by local HuggingFace transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self._pipeline = None
        self._profile = ModelProfile(
            name=model_name,
            provider="huggingface",
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT],
            context_window=2048,
            cost_tier="low",
            local=True,
            default_for=["generate", "reason", "plan", "understand", "structured_generate"],
        )

    def _load_pipeline(self):
        if self._pipeline is None:
            logger.info(f"Loading local HuggingFace model: {self.model_name}")
            from transformers import pipeline
            import torch
            
            # Use CPU for stability if CUDA is not available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                torch_dtype=dtype,
            )
        return self._pipeline

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        start = time.time()
        try:
            pipe = self._load_pipeline()
            
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Limit max tokens for small local models to avoid OOM
            max_new_tokens = min(request.max_tokens, 512)
            
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )
            
            content = outputs[0]["generated_text"][-1]["content"]
            latency = (time.time() - start) * 1000
            
            return IntelligenceResponse(
                content=content.strip(),
                model=self.model_name,
                provider="huggingface",
                latency_ms=latency,
                success=True,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return IntelligenceResponse(
                content="",
                model=self.model_name,
                provider="huggingface",
                latency_ms=latency,
                success=False,
                error=f"HF pipeline error: {str(e)}",
            )

    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        start = time.time()
        schema_class = request.output_schema
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

        system_prompt = (
            f"{request.system_prompt}\n\n"
            f"You MUST respond with valid JSON that conforms to this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Output ONLY the JSON object."
        )

        validation_errors = []

        for attempt in range(1, request.max_retries + 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt},
            ]

            if validation_errors:
                repair_msg = (
                    f"Your previous response had validation errors:\n"
                    f"{validation_errors[-1]}\n\n"
                    f"Fix these errors and output valid JSON."
                )
                messages.append({"role": "user", "content": repair_msg})

            try:
                pipe = self._load_pipeline()
                outputs = pipe(
                    messages,
                    max_new_tokens=512,
                    temperature=request.temperature,
                    do_sample=request.temperature > 0,
                )
                
                raw_content = outputs[0]["generated_text"][-1]["content"]
                
                # Try to extract JSON from response
                start_idx = raw_content.find("{")
                end_idx = raw_content.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        parsed = json.loads(raw_content[start_idx:end_idx])
                    except json.JSONDecodeError as je:
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
                        model=self.model_name,
                        provider="huggingface",
                        attempts=attempt,
                        validation_errors=validation_errors,
                        success=True,
                        latency_ms=latency,
                    )
                except ValidationError as ve:
                    validation_errors.append(f"Attempt {attempt}: Schema validation failed: {str(ve)}")
                    continue
                    
            except Exception as e:
                validation_errors.append(f"Attempt {attempt}: Request failed: {str(e)}")
                continue

        # All retries exhausted
        latency = (time.time() - start) * 1000
        return StructuredGenerationResult(
            raw_response="",
            model=self.model_name,
            provider="huggingface",
            attempts=request.max_retries,
            validation_errors=validation_errors,
            success=False,
            error=f"Structured generation failed after {request.max_retries} attempts",
            latency_ms=latency,
        )

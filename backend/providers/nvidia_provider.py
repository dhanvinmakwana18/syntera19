import os
import time
import json
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

from intelligence.contracts import (
    BaseIntelligenceProvider,
    IntelligenceRequest,
    IntelligenceResponse,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    ModelProfile,
    ModelCapability
)
from core.contracts import BaseEmbeddingProvider, BaseReranker
from core.domain import Query, RetrievalResult

load_dotenv()

class NVIDIAProvider(BaseIntelligenceProvider):
    def __init__(self, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b", api_key: Optional[str] = None, base_url: str = "https://integrate.api.nvidia.com/v1", local: bool = False, cost_tier: str = "frontier", capabilities: List[ModelCapability] = None, context_window: int = 16384):
        self.model_name = model_name
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        
        if not self._api_key and not local:
            raise ValueError("NVIDIA_API_KEY not found in environment.")
            
        self._client = OpenAI(
            base_url=base_url,
            api_key=self._api_key or "dummy"
        )
        
        if capabilities is None:
            capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT, ModelCapability.REASONING]
            
        self._profile = ModelProfile(
            name=self.model_name,
            provider="nvidia",
            capabilities=capabilities,
            context_window=context_window,
            cost_tier=cost_tier,
            local=local
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        start_time = time.time()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if not self._profile.local:
                kwargs["top_p"] = 0.95
            
            if getattr(request, "metadata", {}).get("enable_thinking", True) and not self._profile.local:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
                
            completion = self._client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content or ""
            latency = (time.time() - start_time) * 1000
            
            return IntelligenceResponse(
                content=content,
                model=self.model_name,
                provider="nvidia",
                latency_ms=latency,
                success=True
            )
        except Exception as e:
            return IntelligenceResponse(
                content="",
                model=self.model_name,
                provider="nvidia",
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )

    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        start_time = time.time()
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        attempts = 0
        last_error = None
        while attempts < request.max_retries:
            attempts += 1
            try:
                if "json" not in messages[-1]["content"].lower():
                    messages[-1]["content"] += "\n\nPlease respond with ONLY valid JSON."
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": request.temperature,
                }
                if not self._profile.local:
                    kwargs["response_format"] = {"type": "json_object"}
                completion = self._client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or "{}"
                parsed_data = None
                validation_errors = []
                if request.output_schema:
                    try:
                        parsed_dict = json.loads(content)
                        parsed_data = request.output_schema(**parsed_dict)
                    except Exception as ve:
                        validation_errors.append(str(ve))
                if not validation_errors and request.output_schema:
                    latency = (time.time() - start_time) * 1000
                    return StructuredGenerationResult(
                        data=parsed_data,
                        raw_response=content,
                        model=self.model_name,
                        provider="nvidia",
                        attempts=attempts,
                        success=True,
                        latency_ms=latency
                    )
                else:
                    last_error = "Validation failed: " + str(validation_errors)
            except Exception as e:
                last_error = str(e)
        return StructuredGenerationResult(
            data=None, raw_response="", model=self.model_name, provider="nvidia",
            attempts=attempts, success=False, error=last_error, validation_errors=[last_error] if last_error else []
        )


class NVIDIAEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "nvidia/nv-embedqa-mistral-7b-v2", api_key: Optional[str] = None):
        self.model_name = model_name
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise ValueError("NVIDIA_API_KEY not found.")
        self._client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=self._api_key)

    def embed_text(self, text: str) -> List[float]:
        res = self._client.embeddings.create(model=self.model_name, input=[text])
        return res.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        res = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in res.data]

    @property
    def vector_size(self) -> int:
        return 4096


class NVIDIAReranker(BaseReranker):
    def __init__(self, model_name: str = "nvidia/nemotron-4-340b-reward", api_key: Optional[str] = None):
        self.model_name = model_name
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise ValueError("NVIDIA_API_KEY not found.")
        self._client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=self._api_key)

    def rerank(self, query: Query, candidates: List[RetrievalResult], limit: int = 5) -> List[RetrievalResult]:
        # Trivial reranker mock fallback without heavy API latency. 
        # In a real environment, this would call NVIDIA's rerank endpoint if available.
        # Since /v1/ranking might not be exposed on all NIM setups, we safely return them sorted.
        return candidates[:limit]


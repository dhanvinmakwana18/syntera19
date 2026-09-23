"""
Syntera Intelligence — Model Router

Deterministic model routing policy across the ProviderFabric.
Routes requests based on capability, resources, context length, and task type.
"""
from typing import Dict, List, Optional
import time
from intelligence.contracts import (
    BaseIntelligenceProvider,
    IntelligenceRequest,
    IntelligenceResponse,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    ModelProfile,
    TaskComplexity,
)
from intelligence.fabric import ProviderFabric

class ModelRouter:
    """
    Routes intelligence requests to the best available provider in the Fabric.
    """

    def __init__(self, fabric: Optional[ProviderFabric] = None, default_provider_name: Optional[str] = None, default_provider: Optional[BaseIntelligenceProvider] = None):
        if not fabric:
            fabric = ProviderFabric()
        self.fabric = fabric
        self._default_name = default_provider_name
        
        if default_provider:
            self.fabric.register(default_provider)
            self._default_name = default_provider.profile.name

    def set_default(self, provider_name: str) -> None:
        if not self.fabric.get(provider_name):
            raise ValueError(f"Provider '{provider_name}' not in fabric.")
        self._default_name = provider_name
        
    def register_provider(self, provider: BaseIntelligenceProvider) -> None:
        self.fabric.register(provider)
        if not self._default_name:
            self._default_name = provider.profile.name

    @property
    def available_providers(self) -> List[ModelProfile]:
        return [p.profile for p in self.fabric.all]

    def _estimate_vram_gb(self) -> float:
        # Avoid heavy repeated queries in production. Cache it.
        # Fallback to assumption for RTX 4050 6GB.
        if hasattr(self, '_cached_vram') and (time.time() - getattr(self, '_vram_cache_time', 0) < 60):
            return self._cached_vram
            
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
                self._cached_vram = vram
                self._vram_cache_time = time.time()
                return vram
        except Exception:
            pass
            
        self._cached_vram = 6.0
        self._vram_cache_time = time.time()
        return 6.0

    def route(self, request: IntelligenceRequest) -> BaseIntelligenceProvider:
        # 1. Explicit provider override
        requested = request.metadata.get("provider")
        if requested:
            prov = self.fabric.get(requested)
            if prov:
                return prov

        estimated_tokens = len(request.prompt) / 4
        
        # Handle both IntelligenceRequest and StructuredGenerationRequest safely
        complexity = getattr(request, "complexity", TaskComplexity.MODERATE)
        task_str = getattr(request, "task", "").lower()
        
        is_complex = complexity == TaskComplexity.COMPLEX
        vram_available = self._estimate_vram_gb()
        
        all_providers = self.fabric.all
        local_models = [p for p in all_providers if p.profile.local]
        remote_models = [p for p in all_providers if not p.profile.local]
        
        omni_model = next((p for p in remote_models if "omni" in p.profile.name.lower()), None)
        ultra_model = next((p for p in remote_models if "ultra" in p.profile.name.lower()), None)
        reasoning_remotes = [p for p in remote_models if p.profile.cost_tier in ("high", "frontier")]

        # Agent/Multimodal -> Omni
        if "agent" in task_str or "multimodal" in task_str:
            if omni_model:
                return omni_model

        # Heavy Reasoning -> Ultra
        if is_complex and ultra_model:
            return ultra_model

        # Local Capacity Check (Nemotron 4B local)
        local_capable = False
        local_provider = None
        if local_models:
            lp = local_models[0]
            if estimated_tokens < (lp.profile.context_window * 0.8):
                if vram_available > 2.5: 
                    local_capable = True
                    local_provider = lp

        if local_capable and not is_complex:
            return local_provider

        if reasoning_remotes:
            return reasoning_remotes[0]

        if self._default_name:
            return self.fabric.get(self._default_name)

        if all_providers:
            return all_providers[0]
            
        raise RuntimeError("No intelligence providers available in fabric")

    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        provider = self.route(request)
        return provider.generate(request)

    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        provider = self.route(request)
        return provider.structured_generate(request)

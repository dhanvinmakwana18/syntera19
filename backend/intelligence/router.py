"""
Syntera Intelligence — Model Router

Deterministic model routing policy.
Routes intelligence requests to appropriate providers based on task complexity,
capability requirements, and configuration.
"""
from typing import Dict, List, Optional
from intelligence.contracts import (
    BaseIntelligenceProvider,
    IntelligenceRequest,
    IntelligenceResponse,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    ModelProfile,
    TaskComplexity,
)


class ModelRouter:
    """
    Routes intelligence requests to the best available provider.

    Routing policy:
    1. If a specific provider is requested via metadata, use it.
    2. Match task complexity to model cost tier.
    3. Fall back to default provider.
    """

    def __init__(self, default_provider: Optional[BaseIntelligenceProvider] = None):
        self._providers: Dict[str, BaseIntelligenceProvider] = {}
        self._default: Optional[BaseIntelligenceProvider] = default_provider
        if default_provider:
            self._providers[default_provider.profile.name] = default_provider

    def register_provider(self, provider: BaseIntelligenceProvider) -> None:
        self._providers[provider.profile.name] = provider
        if self._default is None:
            self._default = provider

    def set_default(self, provider_name: str) -> None:
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        self._default = self._providers[provider_name]

    @property
    def available_providers(self) -> List[ModelProfile]:
        return [p.profile for p in self._providers.values()]

    def route(self, request: IntelligenceRequest) -> BaseIntelligenceProvider:
        """Select the best provider for a request."""
        # 1. Explicit provider override
        requested = request.metadata.get("provider")
        if requested and requested in self._providers:
            return self._providers[requested]

        # 2. Task-based routing
        if request.complexity == TaskComplexity.COMPLEX:
            # Prefer frontier/high-tier model
            for p in self._providers.values():
                if p.profile.cost_tier in ("frontier", "high"):
                    return p

        # 3. Default
        if self._default:
            return self._default

        raise RuntimeError("No intelligence providers available")

    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        provider = self.route(request)
        return provider.generate(request)

    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        """Route structured generation to the default provider."""
        if not self._default:
            raise RuntimeError("No intelligence providers available")
        return self._default.structured_generate(request)

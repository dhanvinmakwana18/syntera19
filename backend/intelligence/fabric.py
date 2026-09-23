from typing import Dict, List, Optional
from intelligence.contracts import BaseIntelligenceProvider, ModelProfile

class ProviderFabric:
    """
    An execution/orchestration layer that holds the registered models.
    It simply exposes the available providers for the router to select from.
    """
    def __init__(self):
        self._providers: Dict[str, BaseIntelligenceProvider] = {}
        
    def register(self, provider: BaseIntelligenceProvider) -> None:
        self._providers[provider.profile.name] = provider
        
    def get(self, name: str) -> Optional[BaseIntelligenceProvider]:
        return self._providers.get(name)
        
    @property
    def all(self) -> List[BaseIntelligenceProvider]:
        return list(self._providers.values())

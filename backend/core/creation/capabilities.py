from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Set
from core.creation.domain import CapabilityRequirement

class GraphBlueprint:
    def __init__(self):
        self.nodes: List[Any] = []
        self.edges: List[tuple[str, str]] = []
        self.entry_point: str = None
        self.conditional_edges: List[tuple[str, Any]] = []

class BaseCapability(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        pass
        
    @abstractmethod
    def apply(self, blueprint: GraphBlueprint, container: Any, spec: Any) -> None:
        """Injects nodes and edges into the blueprint based on the specification."""
        pass

class CapabilityRegistry:
    def __init__(self):
        self._capabilities: Dict[str, BaseCapability] = {}
        
    def register(self, capability: BaseCapability) -> None:
        self._capabilities[capability.name] = capability
        
    def get(self, name: str) -> BaseCapability:
        if name not in self._capabilities:
            raise ValueError(f"Capability '{name}' not found in registry.")
        return self._capabilities[name]
        
    def has(self, name: str) -> bool:
        return name in self._capabilities
        
    def resolve_dependencies(self, requested_caps: List[CapabilityRequirement]) -> List[BaseCapability]:
        resolved: Dict[str, BaseCapability] = {}
        
        def resolve_recursive(cap_name: str, stack: Set[str]):
            if cap_name in resolved:
                return
            if cap_name in stack:
                raise ValueError(f"Circular dependency detected for capability '{cap_name}'")
                
            stack.add(cap_name)
            cap = self.get(cap_name)
            
            for dep in cap.dependencies:
                resolve_recursive(dep, stack)
                
            resolved[cap_name] = cap
            stack.remove(cap_name)
            
        for req in requested_caps:
            resolve_recursive(req.name, set())
            
        return list(resolved.values())

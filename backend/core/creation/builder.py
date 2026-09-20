from typing import Any, List, Dict
from core.creation.domain import AISystemSpecification, GeneratedAISystem, EvaluationPlan
from core.creation.capabilities import CapabilityRegistry, GraphBlueprint
from core.graph.executor import ExecutionGraph

class AISystemBuilder:
    def __init__(self, registry: CapabilityRegistry, container: Any):
        self.registry = registry
        self.container = container
        
    def validate_specification(self, spec: AISystemSpecification):
        if not spec.capabilities:
            raise ValueError("System specification must define at least one capability.")
        # Ensure all explicitly requested tools and models resolve conceptually,
        # but the capability application will handle the concrete resolution.

    def build(self, spec: AISystemSpecification) -> GeneratedAISystem:
        self.validate_specification(spec)
        
        # 1. Resolve Capabilities
        ordered_capabilities = self.registry.resolve_dependencies(spec.capabilities)
        
        # 2. Build Graph Blueprint
        blueprint = GraphBlueprint()
        for cap in ordered_capabilities:
            cap.apply(blueprint, self.container, spec)
            
        # 3. Construct ExecutionGraph
        graph = ExecutionGraph()
        
        for node in blueprint.nodes:
            graph.add_node(node)
            
        if blueprint.entry_point:
            graph.set_entry_point(blueprint.entry_point)
            
        for from_node, to_node in blueprint.edges:
            graph.add_edge(from_node, to_node)
            
        for from_node, cond_fn in blueprint.conditional_edges:
            graph.add_conditional_edge(from_node, cond_fn)
            
        # 4. Validate Graph
        try:
            graph.validate()
        except ValueError as e:
            raise ValueError(f"Generated graph is invalid: {e}")
            
        # 5. Create Evaluation Plan
        eval_plan = None
        if spec.evaluation:
            eval_plan = EvaluationPlan(
                system_id=spec.identity.id,
                system_version=spec.identity.version,
                metrics=spec.evaluation.metrics,
                criteria=spec.evaluation.criteria
            )
            
        return GeneratedAISystem(
            specification=spec,
            execution_graph=graph,
            evaluation_plan=eval_plan,
            resolved_capabilities=[c.name for c in ordered_capabilities],
            config={},
            metadata={"status": "GENERATED"}
        )

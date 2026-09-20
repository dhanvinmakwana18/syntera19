from core.creation.capabilities import BaseCapability, GraphBlueprint
from core.creation.domain import AISystemSpecification
from swarm.contracts import SwarmSpec, AgentSpec
from swarm.nodes import SwarmAgentNode

class MultiAgentCapability(BaseCapability):
    @property
    def name(self) -> str:
        return "multi_agent"
        
    @property
    def dependencies(self) -> list[str]:
        return []
        
    def apply(self, blueprint: GraphBlueprint, container, spec: AISystemSpecification):
        # We expect spec.workflow to contain SwarmSpec dictionary
        workflow_data = spec.workflow
        if not workflow_data or "agents" not in workflow_data:
            return
            
        swarm_spec = SwarmSpec(**workflow_data)
        intelligence = container.get_intelligence()
        
        # Add agent nodes
        for agent_spec in swarm_spec.agents:
            node = SwarmAgentNode(agent_spec, intelligence)
            blueprint.nodes.append(node)
            
        # Add edges
        edges = []
        if swarm_spec.workflow_type == "sequential":
            for i in range(len(swarm_spec.agents) - 1):
                edges.append((swarm_spec.agents[i].name, swarm_spec.agents[i+1].name))
            edges.append((swarm_spec.agents[-1].name, "END"))
            
        elif swarm_spec.workflow_type == "custom":
            edges = swarm_spec.edges
            
        # Group edges by from_node to natively support parallelism in ExecutionGraph
        from collections import defaultdict
        grouped_edges = defaultdict(list)
        for from_node, to_node in edges:
            grouped_edges[from_node].append(to_node)
            
        for from_node, to_nodes in grouped_edges.items():
            if len(to_nodes) == 1:
                blueprint.edges.append((from_node, to_nodes[0]))
            else:
                # Use a conditional edge to fork to multiple destinations
                def create_fork(targets):
                    return lambda state: targets
                blueprint.conditional_edges.append((from_node, create_fork(to_nodes)))
                
        if swarm_spec.entry_point:
            blueprint.entry_point = swarm_spec.entry_point
        else:
            blueprint.entry_point = swarm_spec.agents[0].name

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
            built_tools = []
            for t_req in agent_spec.tools:
                if t_req.name == "retrieval":
                    pipe = container.build_pipeline(retrieval_mode="rerank")
                    assembler = container.registry.get_context_assembler("default")
                    from orchestration.agentic.tools import RAGTool
                    built_tools.append(RAGTool(pipe, assembler))
                    
            node = SwarmAgentNode(agent_spec, intelligence, tools=built_tools)
            blueprint.nodes.append(node)
            
        # Add edges
        edges = []
        if swarm_spec.workflow_type == "sequential":
            for i in range(len(swarm_spec.agents) - 1):
                edges.append((swarm_spec.agents[i].name, swarm_spec.agents[i+1].name))
            edges.append((swarm_spec.agents[-1].name, "END"))
            
        elif swarm_spec.workflow_type == "custom":
            edges = swarm_spec.edges
            
        from collections import defaultdict
        grouped_edges_out = defaultdict(list)
        grouped_edges_in = defaultdict(list)
        
        for from_node, to_node in edges:
            grouped_edges_out[from_node].append(to_node)
            grouped_edges_in[to_node].append(from_node)
            
        for to_node, from_nodes in grouped_edges_in.items():
            if to_node != "END" and len(from_nodes) > 1:
                blueprint.dependencies.append((to_node, from_nodes))
            
        for from_node, to_nodes in grouped_edges_out.items():
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
            
        if swarm_spec.failure_policy:
            from core.graph.contracts import FailurePolicy
            try:
                blueprint.failure_policy = FailurePolicy(swarm_spec.failure_policy)
            except ValueError:
                pass
                
        def swarm_lifecycle_observer(event_type: str, kwargs: dict):
            state = kwargs.get("state")
            if not state or not hasattr(state, "events"):
                return
                
            from swarm.events import SwarmEvent, SwarmEventName
            if event_type == "GRAPH_START":
                state.events.append(SwarmEvent(event_name=SwarmEventName.START))
            elif event_type == "GRAPH_COMPLETED":
                state.events.append(SwarmEvent(event_name=SwarmEventName.COMPLETED))
            elif event_type == "GRAPH_FAILED":
                state.events.append(SwarmEvent(event_name=SwarmEventName.FAILED, metadata={"failed_nodes": kwargs.get("failed_nodes")}))
            elif event_type == "NODE_SKIPPED":
                state.events.append(SwarmEvent(event_name=SwarmEventName.AGENT_SKIPPED, agent_id=kwargs.get("node")))
                
        # Attach to blueprint or directly to the graph later. 
        # But GraphBlueprint doesn't have callbacks yet. Let's add it.
        blueprint.callbacks = getattr(blueprint, "callbacks", [])
        blueprint.callbacks.append(swarm_lifecycle_observer)

import pytest
from core.graph.executor import ExecutionGraph
from swarm.contracts import SwarmState, AgentSpec
from swarm.nodes import SwarmAgentNode
from swarm.events import SwarmEventName
from swarm.capabilities import MultiAgentCapability
from core.domain import Query
from core.creation.domain import ModelRequirement, AISystemSpecification, SystemIdentity
from swarm.contracts import SwarmSpec
from core.creation.builder import AISystemBuilder
from core.container import ApplicationContainer
from core.registry import ComponentRegistry

# Mock Intelligence
from intelligence.contracts import IntelligenceResponse
class FakeIntelligence:
    def generate(self, prompt, system_prompt=None, **kwargs):
        return IntelligenceResponse(content="Response", model="fake", provider="fake", success=True, latency_ms=10.0)

def test_swarm_observability():
    intelligence = FakeIntelligence()
    
    agent_a = AgentSpec(name="AgentA", role="Role", system_prompt="You are AgentA.", model=ModelRequirement(task="generate"))
    
    spec = SwarmSpec(
        agents=[agent_a],
        workflow_type="sequential",
        entry_point="AgentA"
    )
    
    sys_spec = AISystemSpecification(
        identity=SystemIdentity(id="1", name="name", version="1"),
        purpose="test",
        capabilities=[],
        workflow=spec.model_dump()
    )
    
    from core.creation.capabilities import GraphBlueprint
    blueprint = GraphBlueprint()
    
    # Normally done in cap.apply, but we simulate it to attach the callbacks
    cap = MultiAgentCapability()
    
    # We need a dummy container
    container = ApplicationContainer()
    container._instances["intelligence"] = intelligence
    
    cap.apply(blueprint, container, sys_spec)
    
    # Add callbacks to graph
    graph = ExecutionGraph()
    graph.callbacks.extend(blueprint.callbacks)
    
    for node in blueprint.nodes:
        graph.add_node(node)
        
    graph.set_entry_point(blueprint.entry_point)
    
    state = SwarmState(query=Query(text="test"))
    graph.run(state)
    
    event_names = [e.event_name for e in state.events]
    
    assert SwarmEventName.START in event_names
    assert SwarmEventName.AGENT_STARTED in event_names
    assert SwarmEventName.MESSAGE_SENT in event_names
    assert SwarmEventName.AGENT_COMPLETED in event_names
    assert SwarmEventName.COMPLETED in event_names

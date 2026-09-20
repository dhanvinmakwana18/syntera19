import pytest
from core.graph.executor import ExecutionGraph
from swarm.contracts import SwarmState, AgentSpec, SwarmSpec
from swarm.nodes import SwarmAgentNode
from core.domain import Query
from core.creation.domain import ModelRequirement
from core.graph.contracts import FailurePolicy
from intelligence.contracts import IntelligenceResponse

class FakeIntelligence:
    def generate(self, prompt, system_prompt=None, **kwargs):
        name = "Unknown"
        if "AgentA" in system_prompt: name = "AgentA"
        elif "AgentB" in system_prompt: name = "AgentB"
        elif "AgentC" in system_prompt: name = "AgentC"
        elif "AgentD" in system_prompt: name = "AgentD"
        elif "FailingAgent" in system_prompt:
            return IntelligenceResponse(content="", model="fake", provider="fake", success=False, error="Simulated failure", latency_ms=10.0)
            
        return IntelligenceResponse(
            content=f"Hello from {name}",
            model="fake",
            provider="fake",
            success=True,
            latency_ms=10.0
        )

def test_swarm_continue_independent():
    intelligence = FakeIntelligence()
    
    agent_a = AgentSpec(name="AgentA", role="Role", system_prompt="You are AgentA.", model=ModelRequirement(task="generate"))
    fail = AgentSpec(name="FailingAgent", role="Role", system_prompt="You are FailingAgent.", model=ModelRequirement(task="generate"))
    agent_b = AgentSpec(name="AgentB", role="Role", system_prompt="You are AgentB.", model=ModelRequirement(task="generate"))
    agent_c = AgentSpec(name="AgentC", role="Role", system_prompt="You are AgentC.", model=ModelRequirement(task="generate"))
    
    graph = ExecutionGraph(failure_policy=FailurePolicy.CONTINUE_INDEPENDENT)
    graph.add_node(SwarmAgentNode(agent_a, intelligence))
    graph.add_node(SwarmAgentNode(fail, intelligence))
    graph.add_node(SwarmAgentNode(agent_b, intelligence))
    graph.add_node(SwarmAgentNode(agent_c, intelligence))
    
    graph.set_entry_point("AgentA")
    # A -> Fail -> B (Dependent branch)
    # A -> C (Independent branch)
    graph.add_conditional_edge("AgentA", lambda s: ["FailingAgent", "AgentC"])
    graph.add_edge("FailingAgent", "AgentB")
    
    state = SwarmState(query=Query(text="test"))
    result = graph.run(state)
    
    # Run fails because one node failed
    assert result.success is False
    
    completed = [t["node"] for t in result.trace if t["status"] == "COMPLETED"]
    assert "AgentA" in completed
    assert "AgentC" in completed  # Independent branch continued!
    assert "AgentB" not in completed # Dependent branch did not execute

def test_swarm_fail_fast():
    intelligence = FakeIntelligence()
    
    agent_a = AgentSpec(name="AgentA", role="Role", system_prompt="You are AgentA.", model=ModelRequirement(task="generate"))
    fail = AgentSpec(name="FailingAgent", role="Role", system_prompt="You are FailingAgent.", model=ModelRequirement(task="generate"))
    agent_b = AgentSpec(name="AgentB", role="Role", system_prompt="You are AgentB.", model=ModelRequirement(task="generate"))
    
    graph = ExecutionGraph(failure_policy=FailurePolicy.FAIL_FAST)
    graph.add_node(SwarmAgentNode(agent_a, intelligence))
    graph.add_node(SwarmAgentNode(fail, intelligence))
    graph.add_node(SwarmAgentNode(agent_b, intelligence))
    
    graph.set_entry_point("AgentA")
    graph.add_edge("AgentA", "FailingAgent")
    graph.add_edge("FailingAgent", "AgentB")
    
    state = SwarmState(query=Query(text="test"))
    result = graph.run(state)
    
    assert result.success is False

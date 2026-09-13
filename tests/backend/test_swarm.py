import pytest
from core.graph.executor import ExecutionGraph
from swarm.contracts import SwarmState, AgentSpec, SwarmSpec
from swarm.nodes import SwarmAgentNode
from core.domain import Query
from intelligence.core import IntelligenceCore
from core.creation.domain import ModelRequirement
from core.graph.contracts import FailurePolicy

# Mock Intelligence
from intelligence.contracts import IntelligenceResponse
class FakeIntelligence:
    def generate(self, prompt, system_prompt=None, **kwargs):
        name = "Unknown"
        if "Researcher" in system_prompt: name = "Researcher"
        elif "Analyst" in system_prompt: name = "Analyst"
        elif "Verifier" in system_prompt: name = "Verifier"
        elif "Writer" in system_prompt: name = "Writer"
        elif "FailingAgent" in system_prompt:
            return IntelligenceResponse(content="", model="fake", provider="fake", success=False, error="Simulated failure", latency_ms=10.0)
            
        return IntelligenceResponse(
            content=f"Hello from {name}",
            model="fake",
            provider="fake",
            success=True,
            latency_ms=10.0
        )

def test_swarm_true_dependencies():
    intelligence = FakeIntelligence()
    
    # 1. Create Agents
    researcher = AgentSpec(name="Researcher", role="Search info", system_prompt="You are Researcher.", model=ModelRequirement(task="generate"))
    analyst = AgentSpec(name="Analyst", role="Analyze text", system_prompt="You are Analyst.", model=ModelRequirement(task="generate"))
    verifier = AgentSpec(name="Verifier", role="Verify", system_prompt="You are Verifier.", model=ModelRequirement(task="generate"))
    writer = AgentSpec(name="Writer", role="Write report", system_prompt="You are Writer.", model=ModelRequirement(task="generate"))
    
    # 2. Add to Graph
    graph = ExecutionGraph(failure_policy=FailurePolicy.FAIL_FAST)
    graph.add_node(SwarmAgentNode(researcher, intelligence))
    graph.add_node(SwarmAgentNode(analyst, intelligence))
    graph.add_node(SwarmAgentNode(verifier, intelligence))
    graph.add_node(SwarmAgentNode(writer, intelligence))
    
    # 3. Setup Edges
    # Researcher -> Analyst
    # Researcher -> Verifier
    # Analyst -> Writer
    # Verifier -> Writer
    graph.set_entry_point("Researcher")
    
    # In capabilities.py, multiple destinations are turned into a conditional edge returning targets
    graph.add_conditional_edge("Researcher", lambda s: ["Analyst", "Verifier"])
    graph.add_edge("Analyst", "Writer")
    graph.add_edge("Verifier", "Writer")
    
    # True Dependency: Writer MUST wait for BOTH Analyst and Verifier
    graph.add_dependency("Writer", ["Analyst", "Verifier"])
    
    # 4. Execute
    state = SwarmState(query=Query(text="Do a multi-agent task"))
    result = graph.run(state)
    
    assert result.success is True
    
    # Verify the order of execution through trace
    nodes_executed = [t["node"] for t in result.trace if t["status"] == "COMPLETED"]
    assert nodes_executed[0] == "Researcher"
    assert set(nodes_executed[1:3]) == {"Analyst", "Verifier"}
    assert nodes_executed[3] == "Writer"
    
def test_swarm_failure_policy_skip_dependents():
    intelligence = FakeIntelligence()
    
    researcher = AgentSpec(name="Researcher", role="Search info", system_prompt="You are Researcher.", model=ModelRequirement(task="generate"))
    failing = AgentSpec(name="FailingAgent", role="Fail", system_prompt="You are FailingAgent.", model=ModelRequirement(task="generate"))
    writer = AgentSpec(name="Writer", role="Write report", system_prompt="You are Writer.", model=ModelRequirement(task="generate"))
    
    graph = ExecutionGraph(failure_policy=FailurePolicy.SKIP_DEPENDENTS)
    graph.add_node(SwarmAgentNode(researcher, intelligence))
    graph.add_node(SwarmAgentNode(failing, intelligence))
    graph.add_node(SwarmAgentNode(writer, intelligence))
    
    graph.set_entry_point("Researcher")
    graph.add_edge("Researcher", "FailingAgent")
    graph.add_edge("FailingAgent", "Writer")
    graph.add_dependency("Writer", ["FailingAgent"])
    
    state = SwarmState(query=Query(text="test skip"))
    result = graph.run(state)
    
    assert result.success is False
    
    completed = [t["node"] for t in result.trace if t["status"] == "COMPLETED"]
    failed = [t["node"] for t in result.trace if t["status"] == "FAILED"]
    skipped = [t["node"] for t in result.trace if t["status"] == "SKIPPED"]
    
    assert "Researcher" in completed
    assert "FailingAgent" in failed
    assert "Writer" in skipped

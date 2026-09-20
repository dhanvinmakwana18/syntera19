import pytest
from core.graph.executor import ExecutionGraph
from swarm.contracts import SwarmState, AgentSpec, SwarmSpec
from swarm.nodes import SwarmAgentNode
from core.domain import Query
from intelligence.core import IntelligenceCore
from core.creation.domain import ModelRequirement

# Mock Intelligence
from intelligence.contracts import IntelligenceResponse
class FakeIntelligence:
    def generate(self, prompt, system_prompt=None, **kwargs):
        # We can extract who we are from the system prompt
        name = "Unknown"
        if "Researcher" in system_prompt: name = "Researcher"
        elif "Analyst_A" in system_prompt: name = "Analyst_A"
        elif "Analyst_B" in system_prompt: name = "Analyst_B"
        elif "Writer" in system_prompt: name = "Writer"
        
        return IntelligenceResponse(
            content=f"Hello from {name}",
            model="fake",
            provider="fake",
            success=True,
            latency_ms=10.0
        )

def test_swarm_parallel_execution():
    intelligence = FakeIntelligence()
    
    # 1. Create Agents
    researcher = AgentSpec(name="Researcher", role="Search info", system_prompt="You are Researcher.", model=ModelRequirement(task="generate"))
    analyst_a = AgentSpec(name="Analyst_A", role="Analyze text", system_prompt="You are Analyst_A.", model=ModelRequirement(task="generate"))
    analyst_b = AgentSpec(name="Analyst_B", role="Analyze data", system_prompt="You are Analyst_B.", model=ModelRequirement(task="generate"))
    writer = AgentSpec(name="Writer", role="Write report", system_prompt="You are Writer.", model=ModelRequirement(task="generate"))
    
    # 2. Add to Graph
    graph = ExecutionGraph()
    graph.add_node(SwarmAgentNode(researcher, intelligence))
    graph.add_node(SwarmAgentNode(analyst_a, intelligence))
    graph.add_node(SwarmAgentNode(analyst_b, intelligence))
    graph.add_node(SwarmAgentNode(writer, intelligence))
    
    # 3. Setup Edges (Fork and Join)
    graph.set_entry_point("Researcher")
    
    # Researcher -> Analyst A and Analyst B (Parallel Fork)
    def researcher_routing(state):
        return ["Analyst_A", "Analyst_B"]
    graph.add_conditional_edge("Researcher", researcher_routing)
    
    # Analyst A -> Writer
    # Analyst B -> Writer
    # This acts as a Join (ExecutionGraph inherently waits for both to finish if they are in the same active_nodes set)
    graph.add_edge("Analyst_A", "Writer")
    graph.add_edge("Analyst_B", "Writer")
    
    # Writer -> END
    graph.add_edge("Writer", "END")
    
    # 4. Execute
    state = SwarmState(query=Query(text="Do a multi-agent task"))
    result = graph.run(state)
    
    assert result.success is True
    assert len(result.final_state.messages) == 4 # Researcher + 2 Analysts + Writer
    
    # Verify the order of execution through trace
    # 1. Researcher
    # 2 & 3. Analyst A & B (order non-deterministic)
    # 4. Writer
    nodes_executed = [t["node"] for t in result.trace]
    assert nodes_executed[0] == "Researcher"
    assert set(nodes_executed[1:3]) == {"Analyst_A", "Analyst_B"}
    assert nodes_executed[3] == "Writer"
    
    # Writer sets final answer
    assert result.final_state.final_answer == "Hello from Writer"

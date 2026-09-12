import pytest
from core.graph.executor import ExecutionGraph
from orchestration.agentic.state import AgentState, AgentPlan, AgentTask
from orchestration.agentic.nodes import PlannerNode, DecisionNode, ToolExecutionNode, CriticNode
from orchestration.agentic.tools import BaseTool, ToolResult
from core.domain import Query, GenerationResult

class FakeLLM:
    def generate(self, prompt, system_prompt, **kwargs):
        if "json" in system_prompt.lower():
            return '{"tasks": [{"id": "1", "description": "test", "tool_name": "fake_tool", "tool_input": {"param": "value"}}]}'
        return "Fake Answer"

class FakeGenerator:
    def generate(self, query, context, **kwargs):
        return GenerationResult(answer="Critic Fake Answer")

class FakeTool(BaseTool):
    @property
    def name(self): return "fake_tool"
    @property
    def description(self): return "Fake description"
    def execute(self, **kwargs):
        return ToolResult(success=True, output="Tool executed", metadata={})

def test_agentic_graph_success():
    graph = ExecutionGraph()
    
    planner = PlannerNode(FakeLLM())
    decision = DecisionNode()
    tool_exec = ToolExecutionNode(tools=[FakeTool()])
    critic = CriticNode(FakeGenerator())
    
    graph.add_node(planner)
    graph.add_node(decision)
    graph.add_node(tool_exec)
    graph.add_node(critic)
    
    graph.set_entry_point("planner")
    # Using dynamic conditional edges for routing loop? No, DecisionNode returns routing hints directly.
    # We still need to register edges to pass graph.validate() 
    graph.add_edge("planner", "decision")
    # We don't add static edges for decision -> tool_execution because validation checks that all edges point to valid nodes.
    # Wait, the validation checks ALL edges in self.edges. Since DecisionNode uses routing hints, we don't need to add it to self.edges!
    graph.add_edge("tool_execution", "decision") # Loop back or just return routing hint? ToolExec returns routing hint to 'decision'
    graph.add_edge("critic", "END")
    
    initial_state = AgentState(query=Query(text="Do something"))
    result = graph.run(initial_state)
    
    assert result.success is True
    assert result.final_state.final_answer == "Critic Fake Answer"
    assert len(result.final_state.observations) == 1
    assert result.final_state.observations[0]["result"] == "Tool executed"

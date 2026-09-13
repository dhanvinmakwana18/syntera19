import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from core.graph.contracts import GraphNode, NodeResult, RoutingDecision
from orchestration.agentic.state import AgentState, AgentPlan, AgentTask
from orchestration.agentic.tools import BaseTool

# We rely on IntelligenceCore for real AI capability
from intelligence.core import IntelligenceCore


class PlannerNode(GraphNode):
    def __init__(self, intelligence: IntelligenceCore):
        self.intelligence = intelligence
        
    @property
    def name(self) -> str:
        return "planner"
        
    def execute(self, state: AgentState) -> NodeResult:
        if state.iteration >= state.max_iterations:
            return NodeResult(state_updates={"error": "Max iterations reached."})
            
        system_prompt = (
            "You are an agentic planner. Break down the user's goal into a sequential plan.\n"
            "The available tool is 'retrieve_documents' (inputs: 'query').\n"
            "Formulate the tasks needed to accomplish the goal."
        )
        prompt = f"Goal: {state.query.text}"
        
        try:
            # We use the robust structured generation from IntelligenceCore
            result = self.intelligence.structured_generate(
                prompt=prompt,
                schema=AgentPlan,
                system_prompt=system_prompt,
                max_retries=3
            )
            
            if not result.success:
                return NodeResult(state_updates={"error": f"Planner failed to generate valid plan: {result.error} (Validation errors: {result.validation_errors})"})
            
            plan = result.data
            return NodeResult(state_updates={"plan": plan, "iteration": state.iteration + 1})
            
        except Exception as e:
            return NodeResult(state_updates={"error": f"Planner encountered an exception: {str(e)}"})

class DecisionNode(GraphNode):
    @property
    def name(self) -> str:
        return "decision"
        
    def execute(self, state: AgentState) -> NodeResult:
        if state.error:
            return NodeResult(routing_decision=RoutingDecision(next_nodes=["END"]))
            
        if not state.plan or state.current_task_idx >= len(state.plan.tasks):
            return NodeResult(routing_decision=RoutingDecision(next_nodes=["critic"]))
            
        return NodeResult(routing_decision=RoutingDecision(next_nodes=["tool_execution"]))

class ToolExecutionNode(GraphNode):
    def __init__(self, tools: List[BaseTool]):
        self.tools = {t.name: t for t in tools}
        
    @property
    def name(self) -> str:
        return "tool_execution"
        
    def execute(self, state: AgentState) -> NodeResult:
        if not state.plan or state.current_task_idx >= len(state.plan.tasks):
            return NodeResult(state_updates={"error": "No task to execute"})
            
        task = state.plan.tasks[state.current_task_idx]
        tool = self.tools.get(task.tool_name)
        
        obs = state.observations.copy()
        
        if not tool:
            task.status = "FAILED"
            task.result = f"Tool {task.tool_name} not found"
            obs.append({"tool": task.tool_name, "result": task.result, "status": "FAILED"})
        else:
            res = tool.execute(**task.tool_input)
            task.status = "COMPLETED" if res.success else "FAILED"
            task.result = res.output
            obs.append({"tool": task.tool_name, "result": res.output, "status": task.status, "metadata": res.metadata})
            
        return NodeResult(state_updates={"observations": obs, "current_task_idx": state.current_task_idx + 1}, routing_decision=RoutingDecision(next_nodes=["decision"]))

class CriticNode(GraphNode):
    def __init__(self, intelligence: IntelligenceCore):
        self.intelligence = intelligence
        
    @property
    def name(self) -> str:
        return "critic"
        
    def execute(self, state: AgentState) -> NodeResult:
        # Critic synthesizes observations into final answer using IntelligenceCore
        context_text = "\n".join([f"Observation from {o['tool']}: {o['result']}" for o in state.observations if o['status'] == "COMPLETED"])
        
        system_prompt = (
            "You are a synthesis critic. Use the provided context observations "
            "to answer the user's goal accurately. Do not hallucinate outside the observations."
        )
        prompt = f"Goal: {state.query.text}\n\nContext:\n{context_text}"
        
        try:
            res = self.intelligence.generate(prompt=prompt, system_prompt=system_prompt)
            if not res.success:
                return NodeResult(state_updates={"error": f"Critic failed to generate answer: {res.error}"})
                
            return NodeResult(state_updates={"final_answer": res.content})
        except Exception as e:
            return NodeResult(state_updates={"error": f"Critic encountered an exception: {e}"})

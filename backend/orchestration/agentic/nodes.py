import json
from typing import Dict, Any, List
from core.graph.contracts import GraphNode, NodeResult, RoutingDecision
from orchestration.agentic.state import AgentState, AgentPlan, AgentTask
from orchestration.agentic.tools import BaseTool

class PlannerNode(GraphNode):
    def __init__(self, llm_provider):
        self.llm = llm_provider
        
    @property
    def name(self) -> str:
        return "planner"
        
    def execute(self, state: AgentState) -> NodeResult:
        if state.iteration >= state.max_iterations:
            return NodeResult(state_updates={"error": "Max iterations reached."})
            
        system_prompt = "You are a planner. The available tool is 'retrieve_documents'. Output JSON: {\"tasks\": [{\"id\": \"t1\", \"description\": \"...\", \"tool_name\": \"retrieve_documents\", \"tool_input\": {\"query\": \"...\"}}]}"
        prompt = f"Goal: {state.query.text}"
        
        try:
            # We mock LLM planning parsing for stability, normally use structured output.
            res = self.llm.generate(prompt=prompt, system_prompt=system_prompt).strip()
            
            # Simple heuristic for safe parsing during architecture hardening phase
            # If the LLM doesn't output JSON cleanly, default to a RAG plan.
            tasks = [AgentTask(id="1", description="Retrieve context", tool_name="retrieve_documents", tool_input={"query": state.query.text})]
            
            if "{" in res and "tasks" in res:
                try:
                    # Very naive parsing
                    parsed = json.loads(res[res.find("{"):res.rfind("}")+1])
                    if "tasks" in parsed:
                        tasks = []
                        for t in parsed["tasks"]:
                            tasks.append(AgentTask(id=t.get("id", "1"), description=t.get("description", ""), tool_name=t.get("tool_name", ""), tool_input=t.get("tool_input", {})))
                except Exception:
                    pass
                    
            plan = AgentPlan(tasks=tasks)
            return NodeResult(state_updates={"plan": plan, "iteration": state.iteration + 1})
            
        except Exception as e:
            return NodeResult(state_updates={"error": f"Planner failed: {str(e)}"})

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
    def __init__(self, generator):
        self.generator = generator
        
    @property
    def name(self) -> str:
        return "critic"
        
    def execute(self, state: AgentState) -> NodeResult:
        # Critic synthesizes observations into final answer
        from core.domain import GenerationContext
        
        context_text = "\n".join([f"Observation from {o['tool']}: {o['result']}" for o in state.observations if o['status'] == "COMPLETED"])
        sources = []
        for o in state.observations:
            if o.get("metadata", {}).get("sources"):
                sources.extend(o["metadata"]["sources"])
                
        context = GenerationContext(text=context_text, sources=sources)
        
        try:
            res = self.generator.generate(state.query, context)
            return NodeResult(state_updates={"final_answer": res.answer})
        except Exception as e:
            return NodeResult(state_updates={"error": f"Critic failed: {e}"})

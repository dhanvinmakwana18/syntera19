import json
from typing import Any
from core.graph.contracts import GraphNode, NodeResult
from swarm.contracts import SwarmState, AgentSpec, SwarmMessage
from intelligence.core import IntelligenceCore
from intelligence.contracts import TaskComplexity

class SwarmAgentNode(GraphNode):
    def __init__(self, spec: AgentSpec, intelligence: IntelligenceCore, tools: List[Any] = None):
        self.spec = spec
        self.intelligence = intelligence
        self.tools = {t.name: t for t in (tools or [])}
        
    @property
    def name(self) -> str:
        return self.spec.name
        
    def execute(self, state: SwarmState) -> NodeResult:
        # Build prompt from state messages and query
        context = f"Goal: {state.query.text}\n\n"
        if state.shared_context:
            context += f"Shared Context: {json.dumps(state.shared_context)}\n\n"
            
        context += "Message History:\n"
        for msg in state.ledger.get_all():
            context += f"From {msg.sender}: {msg.content}\n"
            
        system_prompt = f"You are {self.spec.name}, a {self.spec.role}.\n{self.spec.system_prompt}\n"
        
        # If we have tools, describe them
        if self.tools:
            system_prompt += "\nAvailable Tools:\n"
            for t in self.tools.values():
                system_prompt += f"- {t.name}\n"
                
        system_prompt += "Review the goal and message history, then provide your contribution or response."
        
        from swarm.events import SwarmEvent, SwarmEventName
        
        start_event = SwarmEvent(event_name=SwarmEventName.AGENT_STARTED, agent_id=self.name)
        state.events.append(start_event)
        
        c = TaskComplexity.MODERATE
        if self.spec.model.reasoning_required == "high":
            c = TaskComplexity.COMPLEX
            
        try:
            # We first check if the agent needs to use a tool (simple mock parsing for demo logic)
            # In a robust local LLM, we'd use a Pydantic schema for tool calls.
            # Here we just check if it's the researcher and they have the retrieval tool.
            tool_outputs = []
            if self.tools:
                # Naive execution for all tools for simplicity on CPU
                for t_name, tool in self.tools.items():
                    res = tool.execute(query=state.query.text)
                    if res.success:
                        tool_outputs.append(f"Tool {t_name} output: {res.output}")
                    else:
                        tool_outputs.append(f"Tool {t_name} failed: {res.error}")
                        
                if tool_outputs:
                    context += "\nTool Execution Results:\n" + "\n".join(tool_outputs)
            
            response = self.intelligence.generate(
                prompt=context,
                system_prompt=system_prompt,
                complexity=c
            )
        except Exception as e:
            fail_event = SwarmEvent(event_name=SwarmEventName.AGENT_FAILED, agent_id=self.name, metadata={"error": str(e)})
            state.events.append(fail_event)
            raise e
            
        if not response.success:
            fail_event = SwarmEvent(event_name=SwarmEventName.AGENT_FAILED, agent_id=self.name, metadata={"error": response.error})
            state.events.append(fail_event)
            raise RuntimeError(f"Agent {self.name} failed: {response.error}")
            
        new_msg = SwarmMessage(
            sender=self.name,
            content=response.content,
            metadata={"model": response.model, "latency_ms": response.latency_ms}
        )
        
        # Thread-safe add to the ledger
        state.ledger.add(new_msg)
        
        msg_event = SwarmEvent(event_name=SwarmEventName.MESSAGE_SENT, agent_id=self.name)
        state.events.append(msg_event)
        
        comp_event = SwarmEvent(event_name=SwarmEventName.AGENT_COMPLETED, agent_id=self.name, metadata={"latency_ms": response.latency_ms})
        state.events.append(comp_event)
        
        updates = {}
        if "writer" in self.spec.role.lower() or "writer" in self.name.lower() or "verifier" in self.spec.role.lower():
            updates["final_answer"] = response.content
            
        return NodeResult(state_updates=updates)

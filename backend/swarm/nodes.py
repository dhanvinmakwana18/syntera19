import json
from typing import Any
from core.graph.contracts import GraphNode, NodeResult
from swarm.contracts import SwarmState, AgentSpec, SwarmMessage
from intelligence.core import IntelligenceCore

class SwarmAgentNode(GraphNode):
    def __init__(self, spec: AgentSpec, intelligence: IntelligenceCore):
        self.spec = spec
        self.intelligence = intelligence
        
    @property
    def name(self) -> str:
        return self.spec.name
        
    def execute(self, state: SwarmState) -> NodeResult:
        # Build prompt from state messages and query
        context = f"Goal: {state.query.text}\n\n"
        if state.shared_context:
            context += f"Shared Context: {json.dumps(state.shared_context)}\n\n"
            
        context += "Message History:\n"
        for msg in state.messages:
            context += f"From {msg.sender}: {msg.content}\n"
            
        system_prompt = f"You are {self.spec.name}, a {self.spec.role}.\n{self.spec.system_prompt}\n"
        system_prompt += "Review the goal and message history, then provide your contribution or response."
        
        # We can route complexity based on ModelRequirement, but here we just use generate
        response = self.intelligence.generate(
            prompt=context,
            system_prompt=system_prompt
        )
        
        if not response.success:
            raise RuntimeError(f"Agent {self.name} failed: {response.error}")
            
        new_msg = SwarmMessage(
            sender=self.name,
            content=response.content,
            metadata={"model": response.model, "latency_ms": response.latency_ms}
        )
        
        # Thread-safe append to the list in-place
        state.messages.append(new_msg)
        
        updates = {}
        if "writer" in self.spec.role.lower() or "writer" in self.name.lower() or "verifier" in self.spec.role.lower():
            updates["final_answer"] = response.content
            
        return NodeResult(state_updates=updates)

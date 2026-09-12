# Agentic Graph Architecture (Phase 6)

## Overview
Phase 6 officially unified Syntera's distinct workflows. Previously, standard RAG executed through the ExecutionGraph, while the Agentic loop operated on a separate, hardcoded procedural path within ackend/api/router.py.

This phase successfully ported the entire Agentic loop (Planner, Tools, Critic) into the identical generic DAG Execution Engine introduced in Phase 5C, fulfilling the bridge from a modular RAG pipeline to a universal AI orchestration engine.

## 1. Previous Agentic Architecture & Problems
- **Procedural Trap:** execute_agent() was a monolithic while-loop.
- **Untyped State:** The agent state relied heavily on strings and loose dictionaries.
- **Tool Coupling:** RAG was hardcoded as a direct module dependency within the loop, meaning the agent couldn't dynamically dispatch to an expanding suite of isolated tools.

## 2. New Graph Integration
The Agentic workflow now utilizes the exact same ExecutionGraph as RAG. The API simply invokes container.build_agentic_graph().

### The Agentic Nodes
- **PlannerNode:** Accepts a Query and instructs the configured LLM to synthesize a typed AgentPlan composed of AgentTask instances.
- **DecisionNode:** Acts as a generic router. By evaluating state.current_task_idx, it issues RoutingDecision payloads instructing the graph to dynamically route to either ToolExecutionNode (if tasks remain) or CriticNode (if complete).
- **ToolExecutionNode:** Generically dispatches 	ask.tool_input against isolated BaseTool contracts, capturing observations into the state.
- **CriticNode:** Condenses all successful observations back into a synthesized GenerationContext payload, and invokes the standard generator for the final answer.

## 3. Tool Architecture
Introduced ackend/orchestration/agentic/tools.py.
- **BaseTool Contract:** Forces tools to implement .execute(**kwargs) -> ToolResult.
- **RAG as a Capability:** The previous monolithic RAG execution is now cleanly wrapped as RAGTool. The agent interacts with it purely as a generic interface, receiving semantic context strings back.

## 4. Failure Handling
Agent failures are cleanly partitioned:
- **Tool Failures:** Do not crash the graph. They record a FAILED status in AgentTask, appending the error to observations so the Critic can acknowledge the failure.
- **Planner Loops:** Protected by a max_iterations counter within the PlannerNode.
- **DAG Failures:** The underlying ExecutionGraph safely isolates terminal node crashes.

## 5. Future AI Creation Engine Boundary
This architecture securely establishes the foundation for AI creation. By encapsulating logic within the generic GraphState and GraphNode interfaces, a future system only needs to serialize a JSON definition of Edge logic and Node tools to instantly stamp out new autonomous agents. 

## 6. Future Evolution Engine Boundary
The ExecutionGraph naturally produces an immutable 	race for every execution. The future Evolution Engine will ingest these JSON traces to benchmark agent variants asynchronously, optimizing prompts or routing thresholds without modifying runtime code.

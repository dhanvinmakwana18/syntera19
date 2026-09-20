# Syntera Phase 10: Multi-Agent Intelligence Architecture

## 1. Vision & Core Philosophy
Syntera fundamentally scales from single-agent generation to dynamic Multi-Agent (Swarm) systems. It utilizes the ExecutionGraph natively as an asynchronous map-reduce executor, mapping agents as nodes in a parallel execution topology.

## 2. Multi-Agent Specification
Instead of hardcoding a swarm structure in code, users provide an AISystemSpecification with a workflow dict shaped for MultiAgentCapability. This defines:
- A list of AgentSpecs (each containing a Name, Role, System Prompt, and specific ModelRequirement).
- A workflow_type (e.g., sequential, parallel, custom).
- Exact node routing via edges.

## 3. The Shared Swarm State
A Swarm runs on a unified SwarmState containing:
- query: the user's initial objective.
- messages: a thread-safe list of SwarmMessage objects detailing which agent said what.
- shared_context: cross-agent dictionary memory.
- inal_answer: populated by terminal agents (e.g., Writer or Verifier).

## 4. Native Parallelism & Map-Reduce
Because the ExecutionGraph relies on ThreadPoolExecutor and uses a set-based 
ext_active_nodes approach, it natively acts as a Barrier Synchronization mechanism for Swarm execution:
- **Forking**: If Researcher routes to [Analyst, Verifier], the ExecutionGraph drops them both into the thread pool and awaits both futures simultaneously.
- **Joining**: If Analyst routes to Writer and Verifier routes to Writer, ExecutionGraph naturally de-duplicates the active edge targets. Writer executes exactly once, receiving the aggregated messages state produced by both parallel predecessors.

## 5. Intelligence Core Integration & Routing
Every SwarmAgentNode invokes the IntelligenceCore.generate() method. The IntelligenceCore consults the ModelRouter, enabling granular routing (e.g., Researcher uses a high-context retriever model, while Analyst uses a high-reasoning local model). In the local VM deployment, all agents funnel through the single thread-safe HuggingFaceProvider using the quantized 0.5B instruct model.

## 6. Evaluation and Evolution
Because the entire multi-agent swarm operates inside a standard ExecutionGraph wrapped in a GeneratedAISystem, it inherits 100% compatibility with Phase 8 Evaluation and Phase 9 Evolution. The Evolution Engine can take a single-agent system, generate a Swarm specification instead, build it, evaluate the swarm against the single-agent baseline, and promote it if metrics improve.

## 7. Limitations & Technical Debt
- Thread-safe lists resolve immediate race conditions, but advanced Swarm logic requiring lock mechanisms (e.g., mutating shared JSON files simultaneously) is not yet supported.
- State persistence across process bounds requires serializing the SwarmState to a database which is a future enhancement (Phase 11).

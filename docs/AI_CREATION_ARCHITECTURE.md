# AI Creation Architecture (Phase 7)

## 1. Problem
Previously, Syntera executed systems procedurally or via a hardcoded RAG / Agentic state graph. There was no capability to dynamically *create* novel AI systems from declarative specifications. Syntera needed an architecture that transitions it from merely *running* an AI agent to *generating* and composing AI systems dynamically.

## 2. Architectural Goal
Establish the **AI Creation Foundation**—a deterministic, modular factory layer that ingests a structured requirement (AISystemSpecification), resolves independent capabilities without global state, and returns an inspectable GeneratedAISystem artifact *before* execution.

## 3. AI System Specification
The core declarative contract (ackend/core/creation/domain.py):
- SystemIdentity: Versioned identifier of the AI system.
- CapabilityRequirement: Requests for explicit functional topologies (e.g., "chat", "retrieval", "planning").
- KnowledgeRequirement: Specifications for RAG modes, thresholds, and limits.
- EvaluationRequirement: Encodes explicit evaluation metrics required for the system.
- Provider-agnostic. No references to "Qdrant", "OpenAI", or "LangChain".

## 4. Capability Architecture
Instead of rigid ChatbotAgent or ResearchGraph classes, we built CapabilityRegistry and BaseCapability.
- Capabilities inject nodes and topological edges directly into a GraphBlueprint.
- They are resolved by a dependency-injection stack.
- Example: Requesting ["chat", "retrieval", "verification"] composes a complete Modular RAG stack natively without a pre-existing "RAG Engine" wrapper. Requesting ["planning"] dynamically injects the Planner, Decision, and Tool dispatch loops.

## 5. Construction vs Execution
The AISystemBuilder generates a GeneratedAISystem artifact containing the immutable ExecutionGraph and an EvaluationPlan.
- **Creation:** sys = builder.build(spec)
- **Execution:** sys.execution_graph.run(state)
This enforces a strict boundary. The creation engine will never auto-execute. The generated artifact acts as a deployable footprint.

## 6. Provider Independence & LLM Handling
The creation layer strictly injects domain contracts. Model providers, LLM token limits, and DB instances are instantiated gracefully through the ApplicationContainer and mapped natively to the injected Nodes via capability resolution.

## 7. Future Evolution Engine Boundary
By serializing an AISystemSpecification, the future **Evolution Engine** can parametrically mutate Capabilities, KnowledgeRequirements, or Node routes. It can then spawn Candidate Systems (v2, v3), run them through the extracted EvaluationPlan, and deterministically select the dominant configuration.

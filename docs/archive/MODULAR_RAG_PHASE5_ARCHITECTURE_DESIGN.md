# Syntera Modular RAG: Phase 5 Architecture Design

## 1. Executive Summary
Phase 4 successfully migrated Syntera from a procedural codebase to a true Dependency Injection (DI) architecture with strict domain boundaries for the *retrieval* layer. However, the end-to-end RAG workflow (query processing, context assembly, verification, generation) remains procedurally hardcoded inside API routers and orchestrators. 
Phase 5 aims to extend the modular architecture to the *entire* RAG lifecycle, eliminating hardcoded orchestrators in favor of a flexible, reorderable execution engine. This will transform Syntera from a "Composable Retrieval Engine" into a "True Modular RAG Engine" capable of advanced graph-based workflows, agentic loops, and extensive ablation experimentation.

## 2. Current Architecture Assessment
Currently, Syntera features a `RetrievalPipeline` that executes:
`Retrievers -> Fusion -> Reranker -> PostProcessors`.
This pipeline is clean, composable, and DI-native. 
However, **outside** this pipeline, the architecture reverts to legacy procedural scripts. For example, `api/router.py` explicitly calls `pipeline.run()`, manually passes the result to `ContextBuilder`, manually formats the prompt, and manually calls `llm.generate()`.

**Critical Flaw**: You cannot swap, skip, or inject a "Verifier" step or a "Query Rewriter" step without editing `api/router.py`. The "RAG" execution is not modular; only the "Retrieval" is.

## 3. Existing Modular RAG Capabilities
- **Retrieval Engine**: Fully composable (Dense, Sparse, Hybrid).
- **Fusion**: Abstracted via `BaseFusionStrategy` (RRF implemented).
- **Reranking**: Abstracted via `BaseReranker` (Cross-encoder implemented).
- **Post-Processing**: Abstracted via `BaseNodePostProcessor`.
- **Infrastructure Isolation**: Qdrant, BM25, and SentenceTransformers are isolated behind interfaces (`BaseVectorStore`, `BaseRetriever`, `BaseEmbeddingProvider`).

## 4. Architectural Gaps
1.  **Query Processing**: No abstraction for rewriting, decomposition, or expansion.
2.  **Context Management**: `ContextBuilder` exists but is hardcoded outside the pipeline. It needs to be a composable step.
3.  **Verification**: Exists as standalone functions (`validate_citations`, `groundedness.py`), but cannot be dynamically injected into a pipeline.
4.  **Generation**: No formal abstraction separates Generation from raw LLM prompting in the API route.
5.  **Execution Engine**: No unified engine orchestrates the E2E flow.

---

## 5. Query Processing Architecture
**Responsibility**: Transform the raw user string into one or more optimized `ProcessedQuery` objects before retrieval.
**Classification**: CORE (with OPTIONAL implementations).
**Potential Implementations**:
- `PassthroughQueryProcessor` (Default)
- `RewriteQueryProcessor` (LLM-based)
- `DecompositionQueryProcessor` (For IEG)

## 6. Retrieval Architecture
**Audit**: The current `BaseRetriever` and `RetrievalPipeline` contracts are sufficient. They already support multiple retrievers, parallel execution (via the engine), and independent configuration.
**Gap**: We need to ensure `RetrievalPipeline` can accept a `ProcessedQuery` or list of queries (if decomposed) rather than a single raw string.

## 7. Fusion Architecture
**Audit**: `BaseFusionStrategy` is sufficient. It correctly accepts `List[List[RetrievalResult]]` and operates entirely on domain models. It remains independent of retrieval and reranking.

## 8. Reranking Architecture
**Audit**: `BaseReranker` is sufficient. It accepts `List[RetrievalResult]` and outputs `List[RetrievalResult]`. It natively supports optional execution (the pipeline skips it if None).

## 9. Context Management Architecture
**Responsibility**: Transform `List[RetrievalResult]` into an optimized context payload for generation.
**Design**: Instead of a monolithic `ContextBuilder`, we need a pipeline of `BaseContextProcessor`s:
- `DeduplicatorProcessor`
- `RelevanceFilterProcessor`
- `TokenTruncationProcessor`
- `ContextAssembler` (Final step: Converts nodes to string/structured JSON).

## 10. Verification Architecture
**Responsibility**: Verify the Generation output against the Context.
**Classification**: CORE interface, OPTIONAL execution.
**Design**: 
- `BaseVerifier` contract.
- Implementations: `CitationVerifier`, `GroundednessVerifier`.
- **Output Model**: `VerificationResult(passed: bool, reason: str, metrics: dict)`
If verification fails, the Execution Engine determines the fallback policy (fail request vs. retry).

## 11. Generation Architecture
**Responsibility**: Transform Context + Query into a Final Response.
**Design**: 
- `BaseGenerator` contract.
- Receives a `GenerationRequest` (containing `Query`, `Context`, `SystemPrompt`).
- Returns `GenerationResult` (containing `answer`, `raw_response`, `usage_metrics`).
This isolates the RAG pipeline from the specific `BaseLLM` implementation, allowing specific formatting (e.g. JSON mode, tool calling) to live in the Generator.

---

## 12. Execution Engine / Graph Architecture
**CRITICAL DECISION**: Syntera MUST evolve into a **Generic Execution Engine (DAG/Graph)**.
Hardcoding `Query -> Retrieve -> Generate` in a `ModularRAGPipeline` class defeats true modularity because it prevents loops (Verification -> Fail -> Regenerate) and parallel branches (IEG subqueries).

**Design: The Syntera Graph**
Syntera will adopt a lightweight State-Machine / DAG engine (similar to LangGraph but native, explicitly typed, and lightweight).
- **State**: A shared `RAGState` object (Query, Candidates, Context, Response, VerificationResults).
- **Nodes**: Classes implementing `BaseGraphNode` (e.g., `RetrieveNode`, `GenerateNode`, `VerifyNode`).
- **Edges**: Conditional routing functions determining the next node (e.g., `if verify.passed -> END else -> GenerateNode`).

This generic engine allows:
1. Linear RAG (Standard)
2. Agentic RAG (Retry loops)
3. IEG (Recursive decomposition)

## 13. Domain Data Flow
The flow of typed domain objects through the graph:
1. `QueryRequest` -> **QueryNode** -> `ProcessedQuery`
2. `ProcessedQuery` -> **RetrievalNode** -> `List[RetrievalResult]`
3. `List[RetrievalResult]` -> **ContextNode** -> `GenerationContext`
4. `GenerationContext` + `ProcessedQuery` -> **GenerationNode** -> `GenerationResult`
5. `GenerationResult` + `GenerationContext` -> **VerificationNode** -> `VerificationResult`
6. `GenerationResult` + `VerificationResult` -> **ResponseNode** -> `FinalResponse`

## 14. Module Contracts
New contracts to define in `core/contracts.py`:
- `BaseQueryProcessor`: `def process(query: Query) -> List[ProcessedQuery]`
- `BaseContextProcessor`: `def process(results: List[RetrievalResult]) -> List[RetrievalResult]`
- `BaseContextAssembler`: `def assemble(results: List[RetrievalResult]) -> GenerationContext`
- `BaseGenerator`: `def generate(request: GenerationRequest) -> GenerationResult`
- `BaseVerifier`: `def verify(response: GenerationResult, context: GenerationContext) -> VerificationResult`
- `BaseGraphNode`: `def execute(state: RAGState) -> Dict[str, Any]` (Returns state updates)

## 15. Failure Model
- **Query Processing**: Fallback to original raw query.
- **Retrieval**: Continue if at least one retriever succeeds. Fail pipeline if all fail.
- **Context/Assembly**: Fail fast if output size exceeds strict limits.
- **Generation**: Bubble up provider errors.
- **Verification**: If fail, engine routes to Retry node (up to N times), then gracefully degrades (returns answer but sets `grounded=False`).

## 16. Observability Model
The `RAGState` will maintain a linear append-only `Trace` list. Each `BaseGraphNode` automatically logs its entry, latency, status, and output size into the trace.
- Avoid logging full generation text in trace; log token counts and hashes.

## 17. Configuration Model
A declarative YAML/JSON model mapped to Pydantic:
```yaml
graph_type: "standard_rag" # or "agentic_rag"
nodes:
  query_processor: "passthrough"
  retrievers: ["qdrant", "bm25"]
  fusion: "rrf"
  reranker: "cross_encoder"
  verifier: "groundedness"
  generator: "local_ollama"
max_retries: 2
```
The `ApplicationContainer` parses this configuration to assemble the Graph.

## 18. Component Registry Integration
We will extend `registry.py` to register the new components:
`register_query_processor`, `register_context_assembler`, `register_generator`, `register_verifier`.

## 19. Application Container Integration
`ApplicationContainer` will gain a method `build_graph(config: GraphConfig) -> StateGraph`.
The FastAPI router will simply call `graph = container.build_graph(config)` and `graph.run(initial_state)`.

## 20. Local-First Provider Architecture
`BaseGenerator` implementations will wrap `BaseLLM`. The container will inject a local Ollama `BaseLLM` by default. Verification algorithms (e.g. NLI models) will prefer local CrossEncoders or local LLMs over cloud APIs.

## 21. Evaluation Architecture
Evaluation must target individual Graph Nodes, not just E2E output:
- **Retrieval Node Eval**: Recall@K.
- **Generator Node Eval**: LLM-as-a-judge for style/adherence.
- **Verifier Node Eval**: Precision/Recall of the verifier against a labeled dataset of hallucinations.

## 22. Experimentation / Ablation Architecture
Because the system is a Graph defined by a Pydantic config, A/B testing is trivial. We can define `ConfigA` (with reranker) and `ConfigB` (without reranker), instantiate two graphs, run the same dataset through both, and compare traces and metrics. No code changes required.

## 23. Future GPU/NVIDIA Compatibility
By keeping dependencies inverted via `BaseLLM` and `BaseEmbeddingProvider`, we guarantee that future TensorRT-LLM or vLLM backends can simply be registered in `registry.py` and injected into the Generator/Retriever without modifying the Graph.

## 24. Dependency Direction
`FastAPI Router -> Execution Graph -> Domain Contracts <- Infrastructure Implementations`
The Execution Graph knows about Domain Contracts (`BaseRetriever`, `BaseGenerator`) but NEVER about Qdrant or Ollama.

## 25. Proposed Architecture Diagram
```
[ User Request ]
       │
       ▼
┌───────────────────────────────────────────────┐
│              RAG EXECUTION GRAPH              │
│                                               │
│  [Query Node] ────────► [Retrieval Node]      │
│                              │                │
│                              ▼                │
│  [Generate Node] ◄───── [Context Node]        │
│       │                                       │
│       ▼                                       │
│  [Verify Node] ──(FAIL)──► [Retry Logic]      │
│       │                                       │
│     (PASS)                                    │
│       │                                       │
└───────┼───────────────────────────────────────┘
        ▼
[ Final Response ]
```

## 26. Proposed Execution Flow
1. API receives request.
2. Extracts `GraphConfig` (or uses default).
3. `container.build_graph(config)` creates the DAG.
4. `graph.execute(initial_state)` runs.
5. Node outputs mutate `RAGState`.
6. API formats `RAGState` into JSON response.

## 27. Migration Strategy
1. Define new contracts in `core/contracts.py`.
2. Implement `StateGraph` engine in `core/graph.py`.
3. Wrap existing `RetrievalPipeline` inside a `RetrievalNode`.
4. Wrap existing LLM calls into a `GenerateNode`.
5. Update `api/router.py` to invoke the Graph instead of the hardcoded procedural flow.
6. Delete legacy procedural flow.

## 28. Risks and Tradeoffs
- **Risk**: A Graph engine is inherently more complex to debug than a linear procedural script.
- **Mitigation**: Strict Pydantic state typing and mandatory trace logging at every node boundary.
- **Tradeoff**: Increased memory overhead due to State object copying. (We will use shallow copies or pass-by-reference for large context blocks).

## 29. Explicit Non-Goals
- DO NOT implement Agentic Retry Loops in Phase 5B (only build the Graph that supports them).
- DO NOT implement complex LangChain/LlamaIndex frameworks. Build a native, lightweight Syntera Graph.
- DO NOT rewrite the `RetrievalPipeline` (it will simply be embedded as a Node).

## 30. Phase 5B Implementation Plan
1. **Contracts**: Add `BaseQueryProcessor`, `BaseContextProcessor`, `BaseGenerator`, `BaseVerifier` to `contracts.py`.
2. **State & Graph**: Create `core/graph.py` containing `RAGState`, `BaseGraphNode`, and `ExecutionGraph`.
3. **Nodes**: Implement `RetrievalNode` (wrapping `RetrievalPipeline`), `ContextNode`, `GenerateNode`.
4. **Registry**: Add registration methods for the new contracts.
5. **API Refactor**: Update `api/router.py` to instantiate and run the `ExecutionGraph` instead of procedural logic.
6. **Testing**: Write unit tests for the Graph engine using fake nodes.

## 31. Phase 5 Acceptance Criteria
1. `api/router.py` no longer contains hardcoded prompt formatting or direct LLM generation calls.
2. The entire RAG flow is orchestrated by `ExecutionGraph` executing `BaseGraphNode` instances.
3. Adding a "Verification" step requires zero changes to `router.py` or `GenerateNode`; it only requires adding `VerifyNode` to the Graph definition.

---

# FINAL VERDICT

**PHASE 5A VERDICT:**
**READY FOR IMPLEMENTATION**

Phase 4 successfully eradicated all architectural debt surrounding dependency injection and initialization. The foundation is solid. The sole remaining bottleneck to true modularity is the procedural orchestration in the API and Agent scripts. Evolving to a lightweight Generic Execution Engine (Graph) is the mathematically correct next step to achieve non-linear, experimental Modular RAG.

### Phase 5B Execution Sequence:
1. **Modify**: `backend/core/contracts.py` (Add new Base classes).
2. **Create**: `backend/core/graph.py` (Implement lightweight DAG/State engine).
3. **Create**: `backend/orchestration/nodes/` (Implement RetrieveNode, ContextNode, GenerateNode).
4. **Modify**: `backend/core/registry.py` (Add registry maps for new contracts).
5. **Modify**: `backend/core/container.py` (Add `build_graph()` factory).
6. **Modify**: `backend/api/router.py` (Replace procedural RAG execution with `graph.run()`).
7. **Create/Modify**: `tests/backend/test_graph.py` (Test graph execution and state transitions).

**DO NOT TOUCH**: 
- `backend/retrieval/engine.py` (RetrievalPipeline is fully functional as a sub-component).
- Infrastructure providers (`qdrant_client.py`, `bm25_store.py`).

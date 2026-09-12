# Syntera Modular RAG: Pipeline Architecture (Phase 3)

## 1. Pipeline Architecture
The Syntera pipeline has been transformed from a procedural script into a composable execution engine. The engine itself (`RetrievalPipeline`) contains no implementation-specific business logic. It simply coordinates the execution flow:
```
Query ──► Retrievers ──► Fusion ──► Reranker ──► Post-Processors ──► PipelineResult
```

## 2. Composition Model
Instead of hardcoding "dense" and "sparse" logic, `RetrievalPipeline` accepts:
* `retrievers: List[BaseRetriever]`
* `fusion_strategy: Optional[BaseFusionStrategy]`
* `reranker: Optional[BaseReranker]`
* `post_processors: Optional[List[BaseNodePostProcessor]]`

This allows arbitrary graph construction, satisfying the objective of True Modular RAG.

## 3. Dependency Injection Model
Global singletons are no longer actively constructed by the core pipeline engine. Instead, a Composition Root injects configured implementations into the generic pipeline.

## 4. Composition Root
**Location**: `backend/retrieval/pipeline.py :: build_pipeline()`
This function acts as the application boundary that maps environment configuration (`retrieval_mode`, weights) into actual contract implementations (`QdrantVectorStore`, `BM25Retriever`, `RRFFusionStrategy`), and then yields a fully assembled `RetrievalPipeline`. 

## 5. Retriever Composition
The engine natively runs `N` retrievers. A failure in one retriever is captured in the execution trace and does not crash the pipeline; the engine simply fuses whatever successful candidate lists it obtained.

## 6. Fusion Composition
Fusion is decoupled from the dual "dense/sparse" assumption. `RRFFusionStrategy` dynamically accepts `N` candidate lists. It can be easily swapped out for a different strategy by passing it to the engine.

## 7. Reranker Composition
Reranking is an optional stage injected via `BaseReranker`. The pipeline simply invokes it if present.

## 8. BM25/Qdrant Decoupling
**Architectural fix**: `BM25Store` no longer relies on iterating over Qdrant to build its index at startup. It now strictly owns its own index by dumping/loading from a local disk persistence file (`bm25_index.pkl`). The ingestion workflow (`backend/ingestion/parser.py`) coordinates writing to both stores independently.

## 9. Context Assembler Decoupling
**Architectural fix**: `assembler.py` (now using `ContextBuilder`) is strictly a pure formatting function. It does zero database I/O. Neighbor expansion logic (which requires Qdrant) has been properly abstracted into `NeighborExpansionPostProcessor(BaseNodePostProcessor)`, moving database I/O to the formal post-processing stage.

## 10. Error Isolation
If a retriever fails, the engine logs the failure in its operational trace and continues with the other retrievers. If fusion fails, the engine falls back to the first successful candidate list. All errors are isolated to their stage.

## 11. Observability
`PipelineResult` includes an execution `.trace` containing a structured log of stage latencies, counts, component names, and success/failure statuses.
```json
[
  {"stage": "retrieval_0", "retriever": "VectorStore", "latency": 0.05, "count": 20, "status": "success"},
  {"stage": "retrieval_1", "retriever": "BM25Store", "latency": 0.01, "count": 20, "status": "success"},
  {"stage": "fusion", "strategy": "RRFFusionStrategy", "latency": 0.001, "count": 20, "status": "success"}
]
```

## 12. Compatibility Strategy
The legacy `retrieve_documents` API signature in `pipeline.py` was retained. Internally, it invokes `build_pipeline()` to compose the engine, extracts the results, and wraps them using `ContextBuilder` to match the exact output format expected by upstream API routes.

## 13. Remaining Architectural Debt
* **Container Framework**: The application still lacks a robust IoC container (e.g. `dependency-injector` or FastAPI Depends) for sweeping singleton removal at the API routing layer.
* **Result Serialization**: API endpoints still assume dictionary shapes for certain objects instead of serializing `PipelineResult` via Pydantic directly.

## 14. Phase 4 Recommendations
* Implement a robust application DI container to completely sever ties with the global singletons initialized in `backend/vectorstore/*.py`.
* Expose the pipeline's operational trace to the user/API response for deep telemetry.
* Refactor API routes to rely on pure domain models instead of backward-compatible dictionaries.

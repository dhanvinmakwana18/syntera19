# Syntera Modular RAG: Application Composition & Dependency Architecture (Phase 4 Design)

## 1. Executive Summary
**PHASE 4 DESIGN VERDICT: READY**

Phase 3 successfully isolated the core domain engine from infrastructure implementation. However, the system's "Composition Root" is currently transitional—it maps hardcoded string flags (`"dense"`, `"sparse"`) directly to legacy global singletons at import time. 

Phase 4 will establish a formal Dependency Injection (DI) and Application Lifecycle architecture. It will eliminate import-time global singletons, introduce a typed Component Registry for dynamic configuration, and formalize the FastAPI lifecycle to properly manage expensive resources (Vector DB connections, ML Models).

## 2. Current Architecture Assessment
*   **Composition Root Location**: Currently inside `backend/retrieval/pipeline.py`. It is coupled to both legacy adapter dictionaries and global singletons.
*   **Global Singletons**: Almost every infrastructure file (e.g., `qdrant_client.py`, `bm25_store.py`, `embeddings.py`) instantiates an object at the bottom of the file upon import. This forces the application to load heavy models and establish DB connections the moment a route is imported.
*   **Configuration**: Relies on rigid string matching (`if mode == "dense"`) preventing true plug-and-play extension.
*   **API Boundary**: API routes still handle raw dictionary extraction instead of mapping cleanly from Domain Objects (`PipelineResult`) to HTTP DTOs.

## 3. Global Singleton Inventory
| Singleton | Location | Verdict | Rationale |
| :--- | :--- | :--- | :--- |
| `vector_store` | `qdrant_client.py` | **Must Eliminate** | Eagerly connects to DB on import. Should be managed by lifecycle DI. |
| `bm25_store` | `bm25_store.py` | **Must Eliminate** | Eagerly loads local index on import. Should be lazy/lifecycle-managed. |
| `embedding_provider` | `embeddings.py` | **Must Eliminate** | Eagerly loads HuggingFace models into RAM. |
| `reranker_service` | `reranker.py` | **Must Eliminate** | Eagerly loads CrossEncoders into RAM. |
| `llm_provider` | `llm.py` | **Must Eliminate** | Side-effecting initialization. |
| `settings` | `core/config.py` | **Useful / Keep** | Immutable Pydantic environment configurations are acceptable global state. |

## 4. Composition Root Decision
Syntera should establish a **Central Application Container** at `backend/core/container.py`. 
*   It should NOT use a heavy third-party framework (like `dependency-injector`) to avoid unnecessary cognitive load.
*   It should be a native Python class (`ApplicationContainer`) that knows how to instantiate the registries, providers, and pipeline engines.
*   The FastAPI application will instantiate this container during its startup lifecycle (`lifespan`) and attach it to `app.state.container`.
*   FastAPI routes will use `Depends()` to extract services from `app.state.container`.

## 5. Component Registry Decision
Syntera will implement a lightweight, typed Component Registry (`backend/core/registry.py`).
```python
class ComponentRegistry:
    def register_retriever(self, name: str, factory_callable): ...
    def get_retriever(self, name: str, **kwargs) -> BaseRetriever: ...
```
**Is it a Service Locator?** No. The application components (e.g., `RetrievalPipeline`) will NEVER query the registry. The registry is strictly used by the Composition Root during application startup to resolve configuration strings into injected instances.

## 6. Configuration Architecture
Configuration will evolve from rigid `mode` strings to a declarative pipeline graph:
```yaml
# Conceptual configuration design
pipeline:
  retrievers:
    - type: "qdrant"
    - type: "bm25"
  fusion:
    type: "rrf"
    weights: [0.7, 0.3]
  reranker:
    type: "cross_encoder"
```
Adding a new retriever (e.g., Elasticsearch) will only require:
1. Writing `ElasticsearchRetriever(BaseRetriever)`.
2. Calling `registry.register_retriever("elasticsearch", factory)`.
3. Updating the config YAML/JSON.
*Zero modifications to `RetrievalPipeline` or `pipeline.py` will be needed.*

## 7. Dependency Injection Strategy
We will use **Constructor Injection** exclusively.
*   `RetrievalPipeline(retrievers=[...], fusion_strategy=...)`
*   `DenseRetriever(vector_store=..., embedding_provider=...)`

Dependencies flow down. Components do not construct their dependencies; they receive them.

## 8. Lifecycle Management
Expensive resources will be managed using FastAPI's `lifespan` context manager in `backend/api/main.py`:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Registry
    # 2. Register Implementations
    # 3. Load ML Models (Embeddings, Reranker)
    # 4. Connect to Databases (Qdrant)
    # 5. Build ApplicationContainer
    app.state.container = container
    yield
    # 6. Graceful Shutdown (close connections)
```

## 9. API Boundary Design
**Current**: API route imports `retrieve_documents()`, which returns tuples of strings and legacy dictionaries.
**Target**: 
1. HTTP Request is parsed into `QueryRequestDTO`.
2. Route retrieves `PipelineRunner` from `Depends(get_container)`.
3. `PipelineRunner` returns `PipelineResult` (Domain Object containing `Node`s).
4. Route maps `PipelineResult` to `QueryResponseDTO` using a pure mapper function.

## 10. Dependency Direction
```
API (FastAPI Routes)
 │
 ▼
Application (Composition Root, Registry, Container)
 │
 ▼
Domain (RetrievalPipeline, Domain Models, Contracts)
 ▲
 │
Infrastructure (Qdrant, BM25, SentenceTransformers)
```
**Strict Rule**: API Layer MUST NOT import from Infrastructure Layer. (e.g., `api/router.py` must stop importing `vector_store` for status checks).

## 11. Proposed Architecture Diagram
```
                     config.yaml
                          │
                          ▼
             ┌─────────────────────────┐
             │    Composition Root     │ (FastAPI Lifespan)
             └────────────┬────────────┘
                          │
                  Component Registry
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
     Infra Factories  Retrieval Engine  Application Services
           │              │              │
           ▼              ▼              ▼
      Constructs      Constructs      Constructs
    DenseRetriever  RetrievalPipeline PipelineRunner
           │              │              │
           └──────────────┼──────────────┘
                          ▼
            Stored in app.state.container
                          │
                          ▼
                 FastAPI API Routes (Depends)
```

## 12. Component Construction Flow
1. App boots.
2. `registry.py` is loaded.
3. Infrastructure plugins register their factory functions.
4. `Container` reads configuration.
5. `Container` requests factories from `Registry` to build `EmbeddingProvider`, `QdrantVectorStore`.
6. `Container` builds `RetrievalPipeline`.
7. `Container` builds `PipelineRunner` (Application Service wrapping the pipeline + context formatting).
8. `Container` attached to `app.state`.

## 13. Test Architecture
Because global singletons are gone, testing becomes mathematically pure:
```python
def test_api_with_fake_container():
    app.dependency_overrides[get_container] = get_fake_container
    # get_fake_container returns a Container populated entirely with FakeRetriever, FakeReranker
    response = client.post("/api/v1/chat")
    assert response.status_code == 200
```
This entirely bypasses Qdrant, CUDA, and Localhost LLM dependencies in the test environment, eliminating the flaky timeouts seen in Phase 3.

## 14. Migration Strategy
1. Create `backend/core/registry.py` and `backend/core/container.py`.
2. Strip global singletons from the bottoms of infrastructure files. Replace them with factory functions (e.g., `def create_qdrant_store() -> BaseVectorStore`).
3. Update `backend/api/main.py` to use a `lifespan` hook that populates the container.
4. Rewrite `backend/api/router.py` to use `Depends()` to fetch the pipeline engine instead of `from retrieval.pipeline import retrieve_documents`.
5. Safely delete `backend/retrieval/pipeline.py` (the transitional adapter).

## 15. Risks and Tradeoffs
*   **Risk**: Background tasks or orchestrators (`IEG`, `Agentic`) running outside the FastAPI request lifecycle won't have access to `Depends()`.
*   **Tradeoff/Mitigation**: The `ApplicationContainer` must be accessible outside of HTTP requests. We will provide a bootstrap function (`bootstrap_container()`) that background scripts can call directly.
*   **Risk**: Removing legacy dictionary adapters will require extensive updates to the frontend or API test assertions. 

## 16. Explicit Non-Goals
*   NO third-party DI frameworks (`dependency-injector`, `pinject`). Pure Python is sufficient.
*   NO new retrieval algorithms or LLM fine-tuning.
*   NO distributed infrastructure (Celery/Redis) yet.

## 17. Acceptance Criteria
1. `vector_store = VectorStore()` and similar singletons are physically deleted from module scope.
2. `RetrievalPipeline` receives components via constructor injection.
3. Adding a new retriever requires zero modifications to the core engine or the application container logic.
4. API routes depend exclusively on `app.state.container` and Domain contracts.
5. 100% of the test suite runs instantly without waiting for Ollama/Qdrant connection timeouts because the test container injects pure fakes.

---
**Files to Modify/Create in Phase 4:**
*   `[CREATE]` `backend/core/registry.py`
*   `[CREATE]` `backend/core/container.py`
*   `[MODIFY]` `backend/api/main.py`
*   `[MODIFY]` `backend/api/router.py`
*   `[DELETE]` `backend/retrieval/pipeline.py`
*   `[MODIFY]` `backend/vectorstore/qdrant_client.py` (Delete singleton)
*   `[MODIFY]` `backend/vectorstore/bm25_store.py` (Delete singleton)
*   `[MODIFY]` `backend/providers/embeddings.py` (Delete singleton)
*   `[MODIFY]` `backend/providers/llm.py` (Delete singleton)
*   `[MODIFY]` `backend/retrieval/reranker.py` (Delete singleton)
*   `[MODIFY]` `backend/orchestration/ieg/orchestrator.py`
*   `[MODIFY]` `backend/orchestration/agentic/workflow.py`
*   `[MODIFY]` `tests/backend/*` (Inject FakeContainer)

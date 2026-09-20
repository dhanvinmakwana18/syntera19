# Syntera Technical Design

## 1. Core Architecture Principles

- **Dependency Direction**: High-level application logic (e.g., API routers, Agentic workflows) depends on abstractions (Interfaces/Contracts in `backend/core/`), never directly on low-level implementations (e.g., Qdrant SDK, OpenAI SDK).
- **Provider Abstraction**: All LLM and Embedding interactions are funneled through generic Provider classes. This ensures Syntera can swap between remote (OpenAI) and local (llama.cpp) inference with zero application-layer changes.
- **Model Routing**: `ModelRouter` is designed to be the central brain for cost, latency, and context optimization, decoupling model choice from the business logic.

## 2. Subsystem Designs

### 2.1 ExecutionGraph
Designed as a strictly typed, inspectable Directed Acyclic Graph (DAG). It avoids the "black box" nature of nested Python loops, allowing for:
- Concurrent node execution.
- Real-time observability and event streaming.
- Checkpointing and workflow resumption.

### 2.2 RAG Design
Built as a **Modular Pipeline** rather than a static chain. It implements Reciprocal Rank Fusion (RRF) to seamlessly blend Dense (Vector) and Sparse (BM25) search. A secondary Cross-Encoder reranker ensures high precision before context assembly.

### 2.3 Memory Design
Avoids the anti-pattern of dumping all user interactions into a single massive vector database. Memory is segmented by lifecycle (Working, Episodic, Semantic, Long-Term) using a local relational store, allowing fast pruning and consolidation. It integrates into RAG seamlessly via `MemoryRetriever`.

### 2.4 Local NVIDIA Inference Design
To support execution on consumer hardware (specifically NVIDIA RTX 4050 with 6GB VRAM), the local inference layer is aggressively optimized:
- **Environment**: Strict Python 3.11 isolation to ensure stable CUDA bindings.
- **Runtime**: `llama.cpp` using Python bindings (`llama-cpp-python` with `cu124`), bypassing heavier frameworks like vLLM which require higher VRAM overhead.
- **Models**: GGUF formats at 4-bit quantization (Q4_K_M) strictly targeting <= 8B parameters to prevent OOM system crashes.

---

## 3. Architecture Decision Records (ADRs)

### ADR-001: ExecutionGraph for Orchestration
- **Decision**: Build a custom DAG-based execution engine (`ExecutionGraph`) instead of using raw Python functions or third-party orchestration libraries (like LangChain/Celery).
- **Reason**: We required a system that was deeply inspectable, deterministically resumable, and capable of executing generic nodes without enforcing a specific LLM schema on the user.
- **Alternatives considered**: Celery (too heavy, requires Redis/RabbitMQ), LangGraph (too closely tied to LangChain ecosystem).
- **Consequences**: Higher initial development cost, but total control over durability, checkpointing, and execution state.

### ADR-002: SQLite for Durability & Memory
- **Decision**: Use SQLite for the `DurableStore` and `MemoryStore`.
- **Reason**: Provides transactional safety, zero-setup local storage, and structured querying capabilities without requiring users to run PostgreSQL or Redis via Docker.
- **Alternatives considered**: JSON files (prone to corruption on crash), PostgreSQL (breaks local-first, zero-config requirement).
- **Consequences**: Limits horizontal scaling across multiple servers natively, but perfectly suits Syntera's local-first and swarm deployment targets.

### ADR-003: llama.cpp for Local NVIDIA Inference (Phase 12)
- **Decision**: Utilize `llama.cpp` (via `llama-cpp-python` with CUDA acceleration) for local RTX 4050 inference.
- **Reason**: The target hardware (RTX 4050) has a strict 6GB VRAM limit. `llama.cpp` allows aggressive GGUF quantization (Q4_K_M) with zero-overhead layer offloading, allowing 4B/8B models to run entirely in VRAM at high speeds (~61 tokens/sec).
- **Alternatives considered**: vLLM (failed due to high VRAM overhead for KV cache pre-allocation), HuggingFace Transformers + BitsAndBytes (unstable on Windows/Python 3.14).
- **Consequences**: Restricts local model selection to GGUF format files, requiring a separate download pipeline from standard safetensors.

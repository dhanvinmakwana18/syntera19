# Syntera Product Requirements Document (PRD)

## 1. What is Syntera?
Syntera is an AI engine that creates, runs, evaluates, and evolves AI systems. It is designed as a highly modular, transparent, and durable orchestrator for Retrieval-Augmented Generation (RAG) and Agentic Swarm architectures.

## 2. Core Vision
"Syntera is an AI engine that creates, runs, evaluates, and evolves AI systems."

## 3. Scope & Target Users
**Target Users**: AI engineers, researchers, and enterprises needing a highly deterministic, locally capable, and inspectable AI orchestration framework.
**Use Cases**: Complex RAG, multi-step document reasoning, autonomous workflows, and self-evaluating/evolving system generation.

### 3.1 Explicit Non-Goals
- **AGI / ASI**: Syntera is not attempting to be Artificial General Intelligence or Artificial Superintelligence.
- **Monolithic "Black Box" Execution**: Operations must be observable and measurable.
- **Closed-Ecosystem Lock-in**: Providers (LLMs, Vector Databases) must remain interchangeable.

## 4. Current Capabilities (Implemented)
- **Modular RAG Pipeline**: Structural PDF ingestion, dense (Qdrant) and sparse (BM25) search, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.
- **ExecutionGraph**: A generic, directed acyclic graph (DAG) execution foundation for all agentic and deterministic workflows.
- **Durable Workflows (Phase 11)**: SQLite-backed checkpointing, node-level resume semantics, and configurable retry policies for resilient long-running tasks.
- **Structural Memory (Phase 11)**: Multi-tiered memory (Working, Episodic, Semantic, Long-Term) natively integrated with the retrieval pipeline.
- **Advanced Chunking**: Fixed-token, Sentence-aware, and Structure-aware chunking preserving document hierarchy.
- **Local GPU Inference (Phase 12.3)**: Isolated Python 3.11 environment with CUDA-accelerated `llama.cpp` supporting NVIDIA RTX 4050 local inference natively.

## 5. Intended Future Capabilities (Planned)
- **ModelRouter & NVIDIA Intelligence Layer**: Seamless bridging between remote providers and the local RTX 4050 hardware runtime.
- **AI Creation & Swarm**: Multi-agent orchestration for dynamic task delegation.
- **Coding Agent Capability**: Ability for Syntera to inspect and modify its own repository safely.
- **Evolution Engine**: Automated optimization of prompts, retrieval strategies, and hyperparameters based on measurable evaluation metrics.

## 6. Definition of Syntera v1
Syntera v1 will be achieved when the system can successfully host a multi-agent swarm that utilizes both local (NVIDIA RTX) and remote models via `ModelRouter`, can execute long-running coding/research tasks durably via `ExecutionGraph`, and can demonstrably evaluate and improve a sub-component of itself using the `Evolution Engine`.

## 7. Success Criteria
1. **Performance**: Local inference (e.g., Nemotron 4B) sustains > 50 tokens/sec.
2. **Durability**: 100% recovery of interrupted workflows without duplicating completed node executions.
3. **Accuracy**: RAG pipeline sustains high Recall@K on internal benchmarks.
4. **Evolution**: Automated architectural changes pass the isolated test suite > 90% of the time before promotion.

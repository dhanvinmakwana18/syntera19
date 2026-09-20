# Syntera Living Roadmap

## COMPLETED
- **Phase 1-4**: Core FastAPI Backend, LLM Provider Abstractions, Initial RAG.
- **Phase 5**: ExecutionGraph Framework (Custom DAG orchestrator).
- **Phase 6-10**: Agentic Node Abstractions, Swarm foundations, Codebase Hardening.
- **Phase 11**: Durable Workflows, Memory & Advanced Chunking.
  - Checkpointing at node batch boundaries.
  - SQLite persistence layer for resilience.
  - Multi-tiered Memory (Working, Episodic, Semantic).
  - Structure-Aware and Semantic Chunkers.
- **Phase 12.1 - 12.3 (Task 2)**: NVIDIA Native Intelligence Layer MVP.
  - Python 3.11 isolated environment (`.syntera-nvidia`).
  - CUDA-enabled PyTorch environment setup.
  - RTX 4050 CUDA verification passed.
  - `llama.cpp` CUDA runtime compiled and verified.
  - Downloaded and successfully executed NVIDIA Nemotron 4B GGUF Q4_K_M entirely on GPU.
  - Benchmark achieved ~61 tokens/sec across 3 successful runs.

## IN PROGRESS
- **Phase 12 (Ongoing)**: NVIDIA API provider & ModelRouter integration.
  - Wrapping the verified `llama.cpp` local runtime into a standard Syntera `LocalGGUFProvider`.
  - Integrating `ModelRouter` to allow IntelligenceCore to seamlessly switch between local inference and external APIs.

## NEXT
- **Coding-Agent Capability**: 
  - Equip agent nodes with tools to inspect the Syntera repository.
  - Enable safe code generation and modification.
  - Integrate test execution pipelines for agents.
- **Controlled Code Evolution**:
  - Implement the Evolution Engine to automatically evaluate and optimize architecture.

## FUTURE
- **Full Swarm Orchestration**: Independent agents collaborating and sharing sub-graphs dynamically.
- **Advanced RAG Modalities**: Image and multimodal context ingestion.
- **Dynamic Context Scaling**: Automatic scaling of `n_ctx` based on hardware availability.

## BLOCKED / NEEDS RESEARCH
- **Python 3.14 Support for Local Inference**: Python 3.14 currently lacks stable C++ compiler ecosystem support on Windows for AI libraries (like `bitsandbytes`). We are currently pinned to the isolated `.syntera-nvidia` Python 3.11 environment for local inference.

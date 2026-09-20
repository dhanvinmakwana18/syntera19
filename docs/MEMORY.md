# Syntera Project Memory

This document serves as persistent project knowledge to prevent repeated rediscovery of context, hardware limitations, and past architectural decisions by developers and future AI coding agents.

## 1. Hardware Constraints & Runtimes
- **Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU.
- **VRAM Constraint**: 6 GB (effectively ~5.8 GB usable).
- **RAM Constraint**: 16 GB Total (often ~2.5 GB available during heavy OS use), strictly limiting safe CPU offloading.
- **Consequence**: Models must remain <= 8B parameters. We heavily rely on 4-bit quantization (Q4_K_M) via GGUF files to fit weights and KV cache entirely inside the 6GB VRAM ceiling.
- **Runtime Choice**: `llama.cpp` (via `llama-cpp-python` `cu124` wheels) is the proven local engine. `vLLM` is explicitly rejected due to its aggressive VRAM pre-allocation causing OOMs on 6GB GPUs.

## 2. Benchmark Baselines (Phase 12.3)
- **Model**: `Nemotron-Mini-4B-Instruct-Q4_K_M.gguf`
- **Engine**: `llama-cpp-python` (CUDA 12.4 backend)
- **Speed**: ~61 tokens/second.
- **Load Time**: ~4.9 seconds.
- **VRAM Usage**: ~3.0 GB (Full GPU offload).

## 3. Important Technical Lessons
1. **Python 3.14 Compatibility**: The global system is running bleeding-edge Python 3.14, which breaks standard AI package compilation on Windows (especially `bitsandbytes` and `torch`). 
   - *Solution*: A strict, isolated virtual environment (`.syntera-nvidia/`) using Python 3.11 was created exclusively for local inference. Do not attempt to run local AI workloads in the global 3.14 environment.
2. **CUDA DLL Loading on Windows**: Precompiled `llama-cpp-python` CUDA wheels often fail to find `llama.dll` dependencies on Windows if the PyTorch `lib` directory is not injected into the `PATH` and `os.add_dll_directory` prior to import. This fix is implemented in our benchmark scripts.
3. **ExecutionGraph Failures**: ThreadPool executors handling `FAIL_FAST` must calculate `pending_nodes` explicitly by observing abandoned sibling threads, otherwise graph resumption loses track of interrupted executions.

## 4. Architectural Rules of Thumb
- **Model Routing**: The application layer NEVER instantiates an LLM directly. All calls flow through generic provider contracts, allowing `ModelRouter` to swap in the RTX 4050 backend seamlessly.
- **ExecutionGraph**: If a process requires more than one LLM call or API step, it belongs in a Graph Node, not a nested Python loop.
- **SQLite over Monoliths**: Durability and Memory rely on SQLite. We explicitly reject requiring users to run Docker/Redis/Postgres for basic agent functionality.

## 5. Terminology
- **ExecutionGraph**: The generic DAG execution engine.
- **IntelligenceCore**: The abstract interface for model reasoning.
- **ModelRouter**: The traffic controller deciding between remote APIs and local CUDA inference.
- **Evolution Engine**: The subsystem responsible for self-measuring and mutating prompts/code.
- **DurableStore**: The SQLite persistence layer enabling workflow pause/resume.

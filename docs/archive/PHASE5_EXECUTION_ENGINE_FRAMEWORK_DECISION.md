# Phase 5 Execution Engine Framework Decision

## Overview
Phase 5 requires a generic execution engine to orchestrate Modular RAG workflows, supporting non-linear execution, conditional routing, and future agentic loops. We evaluated three architectural approaches:
1. Native Python State Graph
2. LangGraph
3. LangChain + LangGraph

## Evaluation

### 1. LangChain + LangGraph
*   **Pros**: Massive ecosystem of integrations, off-the-shelf tools, and built-in memory/checkpointing.
*   **Cons**: Introduces massive dependency bloat. LangChain relies heavily on implicit state passing (`RunnablePassthrough`), proprietary types (`Document`, `AIMessage`), and frequent breaking API changes. Integrating it would require abandoning or wrapping Syntera's existing strict domain contracts (`Node`, `RetrievalResult`) and DI container (`ApplicationContainer`), fundamentally violating Syntera's architectural principles.
*   **Verdict**: REJECTED.

### 2. LangGraph (Standalone)
*   **Pros**: Excellent abstraction for cyclic graphs. Provides built-in checkpointing (SQLite/Postgres) which is highly valuable for persistent autonomous agents.
*   **Cons**: Even when used without the broader LangChain ecosystem, LangGraph heavily utilizes `langchain-core` types (like `BaseMessage`). Its execution model relies on channels and reducers which introduces cognitive overhead and abstraction leaks when dealing with simple sequential or branching pipelines. Furthermore, debugging asynchronous state reducers can be notoriously difficult.
*   **Verdict**: REJECTED.

### 3. Native Python State Graph
*   **Pros**: 
    *   **Zero Framework Lock-in**: Complete control over the execution loop.
    *   **Strict Typing**: We can use pure Pydantic for the `RAGState`, preventing uncontrolled mutation.
    *   **DI Alignment**: Perfectly integrates with our `ComponentRegistry` and `ApplicationContainer`.
    *   **Debuggability**: A native while-loop graph executor is trivial to step through with a debugger.
    *   **Lightweight**: Can be implemented in ~150 lines of robust code.
*   **Cons**: We must implement checkpointing/durability ourselves if we need persistent agents in the future.
*   **Verdict**: ACCEPTED.

## Final Decision
**Syntera will implement a Native Python State Graph.** 
The graph engine will consist of `BaseGraphNode` contracts and a central `ExecutionGraph` runner. The engine will read and mutate a `RAGState` Pydantic model. This preserves the dependency inversion principles established in Phase 4 and keeps Syntera lean, explicitly typed, and engineering-focused.

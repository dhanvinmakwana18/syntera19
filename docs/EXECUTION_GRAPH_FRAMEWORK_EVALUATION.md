# Execution Graph Framework Evaluation

## Overview
As part of Phase 5C, we evaluated adopting a third-party framework (LangGraph, LangChain) versus hardening our native Syntera DAG Execution Engine to orchestrate RAG, Agentic, and future Evolution workflows.

## Candidates

1. **Native Syntera Graph Engine** (Selected)
2. **LangGraph** (Evaluated)
3. **LangChain + LangGraph** (Evaluated)

## Evaluation Criteria

### 1. State Model and Typing
- **LangChain**: Relies heavily on weakly typed generic dicts (RunnablePassthrough, kwargs tunneling) which contradicts Syntera's strict typed domain contracts (RAGState, Query, NodeResult).
- **LangGraph**: Strongly typed state is supported via TypedDict or Pydantic, but mutation relies on implicit reducers (e.g., operator.add). 
- **Syntera Native**: Uses explicit Pydantic domains (GraphState) mutated predictably via structured NodeResult payloads (state_updates, outing_decision). Complete type safety natively integrated with our Phase 2 contracts.

### 2. Branching & Parallelism (Fan-out/Fan-in)
- **LangGraph**: Supports conditional routing through branching edges, and fan-out/fan-in via the new Send API (Map-Reduce) or multiple parallel edges.
- **Syntera Native**: Natively supports conditional branching and executes active branches concurrently using Python's built-in ThreadPoolExecutor. State merge handles fan-in elegantly without requiring specialized mapping abstractions.

### 3. Observability and Telemetry
- **LangChain/LangGraph**: Imposes LangSmith integration for observability, which requires external vendor lock-in or running local infrastructure.
- **Syntera Native**: Telemetry is inherently integrated. The executor captures 
ode, status, latency, ttempts, and updates automatically into an accessible, immutable 	race array inside the state payload, allowing the API router to output zero-friction JSON traces.

### 4. Failure Isolation and Retries
- **LangGraph**: Retries are supported but often complex to configure at the node level without relying on the underlying LCEL (LangChain Expression Language) with_retry() wrapper.
- **Syntera Native**: The Execution Engine natively wraps node executions, offering explicit max_retries, configurable fallbacks, and localized error capturing within the trace without crashing the entire DAG.

### 5. Dependency Cost and Lock-in
- **LangChain/LangGraph**: Introduces massive dependency trees, frequent breaking changes, and a highly opinionated ecosystem that forces architecture to bend to its Runnable interface.
- **Syntera Native**: Zero external dependencies beyond pydantic. Complete architectural ownership.

## Decision: Native Syntera Execution Graph

**Verdict:** We explicitly reject LangChain and LangGraph.

Syntera requires a robust, generic DAG executor capable of executing agentic loops and autonomous jobs. However, importing LangGraph introduces unacceptable framework lock-in that overrides our domain contracts. 

By hardening the native ExecutionGraph, we achieved 100% of the required DAG capabilities—sequential execution, conditional routing, parallel fan-out, and failure isolation—in under 150 lines of clean, readable Python code, natively integrated with our existing ApplicationContainer dependency injection.

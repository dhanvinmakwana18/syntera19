# Modular RAG Execution Engine

## 1. Architecture Overview
The Syntera Modular RAG Execution Engine is a lightweight, generic State Graph (DAG) built in native Python. It replaces hardcoded procedural pipelines with a declarative graph of nodes and conditional edges.

## 2. Core Components

### 2.1 State Model (`RAGState`)
The `RAGState` is a strictly typed Pydantic model representing the memory of the execution graph. It is passed to every node.
```python
class RAGState(BaseModel):
    query: Query
    processed_queries: List[Query] = []
    retrieval_results: List[RetrievalResult] = []
    context: GenerationContext = None
    generation_result: GenerationResult = None
    verification_result: VerificationResult = None
    metadata: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
```

### 2.2 Node Contract (`BaseGraphNode`)
A node performs a single responsibility (e.g., retrieving documents, generating an answer).
```python
class BaseGraphNode(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @abstractmethod
    def execute(self, state: RAGState) -> Dict[str, Any]: 
        pass
```
Nodes return a dictionary of state updates, which the execution engine merges into the `RAGState`.

### 2.3 The Execution Graph
The `ExecutionGraph` maintains a registry of nodes and edges.
*   **Edges**: `add_edge(from_node: str, to_node: str)`
*   **Conditional Edges**: `add_conditional_edge(from_node: str, condition_fn: Callable[[RAGState], str])`

The `run()` method operates a while-loop, transitioning from the `START` node to the `END` node.

## 3. Module Contracts
The graph wraps domain-specific contracts:
*   `BaseQueryProcessor`: Processes the raw query (e.g., rewriting).
*   `BaseContextProcessor`: Modular context operations (deduplication, truncation).
*   `BaseGenerator`: RAG-specific prompt handling and generation, decoupled from `BaseLLM`.
*   `BaseVerifier`: Verifies output groundedness and citation validity.

## 4. Execution Semantics
1. The initial query initializes `RAGState`.
2. The engine evaluates the current node, calling `node.execute(state)`.
3. The engine updates `RAGState` with the returned dictionary.
4. The engine appends a trace entry (latency, node name, status).
5. The engine resolves the next node via edges or conditional edges.
6. Execution terminates when the `END` node is reached or an unhandled exception occurs.

## 5. Observability
Every execution yields an append-only `trace` list inside the state. Nodes do not need to manually manage the trace; the `ExecutionGraph` automatically logs node transitions, latencies, and output shapes.

## 6. Configuration
The graph topology is configured programmatically (and later via YAML) via the `ApplicationContainer`, which constructs the nodes using the `ComponentRegistry` and wires the edges.

# Execution Graph Hardening (Phase 5C)

## 1. Previous Architecture and Problems Discovered
The Phase 5B architecture introduced the ExecutionGraph, but it was functionally a linear sequence executor wrapping the RAG steps. It had the following limitations:
- **Simple Edges:** The graph executor only supported a flat queue of active nodes without genuine fan-out capabilities.
- **Generic State Mutation:** Mutations were done by returning a raw Dict[str, Any] which bypassed strict domain contracts and intent.
- **Global Registry State:** ComponentRegistry utilized a global instance (egistry), which ApplicationContainer and modules implicitly accessed. This broke pure Dependency Injection.
- **Monolithic Context Assembler:** ContextBuilder performed filtering, grouping, sorting, deduplication, and formatting inside a single method, which tightly coupled independent behaviors.

## 2. New Architecture
The graph engine has been rewritten into a Bulk Synchronous Parallel (BSP) executor that uses explicit concurrency and state boundaries.

### Graph Contracts
Introduced ackend/core/graph/contracts.py:
- GraphState: Base Pydantic model for valid state injection.
- NodeResult: Typed payload returned by nodes consisting of state_updates, outing_decision, and metrics.
- RoutingDecision: Typed structure dictating 
ext_nodes.
- GraphNode: Explicit execute method signature.

### State Model & Controlled Transitions
The graph executor no longer accepts arbitrary dictionaries. State transitions strictly enforce that a NodeResult updates properties natively on the GraphState (enforced via Pydantic).

### Fan-out / Fan-in & Concurrency
The executor tracks ctive_nodes at each tick and executes them concurrently using concurrent.futures.ThreadPoolExecutor. 
- **Fan-out:** If a node or conditional edge returns multiple destinations, they run in parallel in the next superstep.
- **Fan-in:** ExecutionGraph automatically deduplicates identical destinations within a single superstep, implicitly synchronizing graph branches that converge.

### Failure Handling
ExecutionGraph._execute_node_with_retry encapsulates nodes. By default, it retries failures. If it terminally fails, it catches the exception and returns a structured GraphExecutionResult(success=False) containing the trace, isolating branch failures from crashing the orchestration API.

### Context Architecture
The monolithic ContextBuilder was dismantled into PipelineContextAssembler, which is now a composable pipeline taking independent interfaces:
- ContextFilter (e.g., ThresholdFilter)
- ContextSorter (e.g., GroupAndRankSorter)
- ContextFormatter (e.g., StandardFormatter)

### Dependency Injection (Zero Global State)
Global registry logic was eradicated. The ComponentRegistry is now instantiated explicitly within uild_container() and passed cleanly into modules via .register(registry) hooks. The executor uses ApplicationContainer purely, with no global state discovery.

## 3. Testing Strategy
- Tests were heavily expanded in 	ests/backend/.
- Fakes (FakeQueryProcessor, etc.) rigorously test graph behavior completely isolated from RAG logic.
- Tested features include conditional routing, node success, parallel tracing, state boundaries, and pipeline composition.

## 4. Extension Points
- Graph supports attaching custom nodes via DI container without modifying the ExecutionGraph class.
- Reusable for future Agentic logic (e.g., ToolNode, CriticNode) which will only need to accept and mutate a typed GraphState.

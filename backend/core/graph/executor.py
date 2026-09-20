import time
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.graph.contracts import GraphNode, GraphState, GraphExecutionResult, NodeResult, RoutingDecision, FailurePolicy, NodeStatus
from core.durability.contracts import DurableStore, Checkpoint, WorkflowStatus, RetryPolicy, FailureRecord

logger = logging.getLogger(__name__)

class ExecutionGraph:
    def __init__(self, max_workers: int = 4, failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST, workflow_id: Optional[str] = None, store: Optional[DurableStore] = None, retry_policy: Optional[RetryPolicy] = None):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable[[Any], List[str]]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.entry_point: str = ""
        self.max_workers = max_workers
        self.failure_policy = failure_policy
        self.callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.workflow_id = workflow_id
        self.store = store
        self.retry_policy = retry_policy or RetryPolicy()

    def add_node(self, node: GraphNode):
        self.nodes[node.name] = node

    def set_entry_point(self, node_name: str):
        self.entry_point = node_name

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edge(self, from_node: str, condition_fn: Callable[[Any], List[str]]):
        self.conditional_edges[from_node] = condition_fn
        
    def add_dependency(self, node: str, depends_on: List[str]):
        if node not in self.dependencies:
            self.dependencies[node] = []
        self.dependencies[node].extend(depends_on)
        
    def validate(self):
        if not self.entry_point:
            raise ValueError("No entry point set.")
        if self.entry_point not in self.nodes:
            raise ValueError(f"Entry point '{self.entry_point}' is not a registered node.")
            
        for from_node, to_node in self.edges.items():
            if from_node not in self.nodes:
                raise ValueError(f"Edge origin '{from_node}' is not a registered node.")
            if to_node != "END" and to_node not in self.nodes:
                raise ValueError(f"Edge destination '{to_node}' is not a registered node.")

    def _fire_event(self, event_type: str, **kwargs):
        for cb in self.callbacks:
            try:
                cb(event_type, kwargs)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def run(self, state: GraphState) -> GraphExecutionResult:
        self.validate()
        self._fire_event("GRAPH_START", state=state)
        
        if self.store and self.workflow_id:
            checkpoint = Checkpoint(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.RUNNING,
                state_data=state.model_dump(),
                completed_nodes=[],
                failed_nodes=[],
                active_nodes=[self.entry_point],
                trace_so_far=[]
            )
            self.store.save_checkpoint(checkpoint)
            self._fire_event("CHECKPOINT_SAVED", workflow_id=self.workflow_id, checkpoint=checkpoint)
            
        return self._run_loop(state, set([self.entry_point]), set(), set(), [])

    def resume(self, state_class: Type[GraphState]) -> GraphExecutionResult:
        if not self.store or not self.workflow_id:
            raise ValueError("Cannot resume without store and workflow_id")
            
        checkpoint = self.store.get_checkpoint(self.workflow_id)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for workflow {self.workflow_id}")
            
        state = state_class.model_validate(checkpoint.state_data)
        self._fire_event("WORKFLOW_RESUMED", workflow_id=self.workflow_id)
        
        return self._run_loop(
            state=state,
            active_nodes=set(checkpoint.active_nodes),
            completed_nodes=set(checkpoint.completed_nodes),
            failed_nodes=set(checkpoint.failed_nodes),
            trace=checkpoint.trace_so_far
        )

    def _run_loop(self, state: GraphState, active_nodes: Set[str], completed_nodes: Set[str], failed_nodes: Set[str], trace: List[Dict[str, Any]]) -> GraphExecutionResult:
        blocked_nodes = set()
        skipped_nodes = set()
        
        while active_nodes or blocked_nodes:
            unblocked = set()
            for n in blocked_nodes:
                deps = self.dependencies.get(n, [])
                
                if any(dep in skipped_nodes or dep in failed_nodes for dep in deps):
                    if self.failure_policy == FailurePolicy.SKIP_DEPENDENTS:
                        logger.warning(f"Skipping {n} because a dependency failed or was skipped.")
                        skipped_nodes.add(n)
                        trace.append({"node": n, "status": NodeStatus.SKIPPED.value, "reason": "dependency failed/skipped"})
                        self._fire_event("NODE_SKIPPED", state=state, node=n)
                        unblocked.add(n)
                        continue
                        
                if all(dep in completed_nodes for dep in deps):
                    unblocked.add(n)
                    
            blocked_nodes.difference_update(unblocked)
            active_nodes.update(n for n in unblocked if n not in skipped_nodes)
            
            ready_to_run = set()
            for n in active_nodes:
                deps = self.dependencies.get(n, [])
                if any(dep in skipped_nodes or dep in failed_nodes for dep in deps) and self.failure_policy == FailurePolicy.SKIP_DEPENDENTS:
                    skipped_nodes.add(n)
                    trace.append({"node": n, "status": NodeStatus.SKIPPED.value, "reason": "dependency failed/skipped"})
                    continue
                    
                if all(dep in completed_nodes for dep in deps):
                    ready_to_run.add(n)
                else:
                    blocked_nodes.add(n)
                    
            active_nodes = ready_to_run
            
            if not active_nodes and blocked_nodes:
                if self.failure_policy == FailurePolicy.CONTINUE_INDEPENDENT:
                    logger.warning(f"Nodes permanently blocked due to failure/deadlock: {blocked_nodes}")
                    break
                else:
                    raise RuntimeError(f"Deadlock detected! Blocked nodes: {blocked_nodes}")
                    
            if not active_nodes:
                break
                
            next_active_nodes = set()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_node = {
                    executor.submit(self._execute_node_with_retry, self.nodes[node_name], state): node_name
                    for node_name in active_nodes if node_name != "END"
                }
                
                for future in as_completed(future_to_node):
                    node_name = future_to_node[future]
                    try:
                        node_result, latency, attempts = future.result()
                        
                        for k, v in node_result.state_updates.items():
                            if hasattr(state, k):
                                setattr(state, k, v)
                                
                        trace.append({
                            "node": node_name,
                            "status": NodeStatus.COMPLETED.value,
                            "latency": latency,
                            "attempts": attempts,
                            "updates": list(node_result.state_updates.keys())
                        })
                        
                        completed_nodes.add(node_name)
                        
                        destinations = []
                        if node_result.routing_decision:
                            destinations = node_result.routing_decision.next_nodes
                        elif node_name in self.conditional_edges:
                            destinations = self.conditional_edges[node_name](state)
                        elif node_name in self.edges:
                            destinations = [self.edges[node_name]]
                        else:
                            destinations = ["END"]
                            
                        for dest in destinations:
                            if dest != "END" and dest not in failed_nodes:
                                next_active_nodes.add(dest)
                                
                    except Exception as e:
                        logger.error(f"Node {node_name} failed: {e}")
                        failed_nodes.add(node_name)
                        trace.append({
                            "node": node_name,
                            "status": NodeStatus.FAILED.value,
                            "error": str(e)
                        })
                        
                        if self.failure_policy == FailurePolicy.FAIL_FAST:
                            self._fire_event("GRAPH_FAILED", state=state, failed_nodes=list(failed_nodes))
                            
                            # Log checkpoint on FAIL_FAST before returning
                            if self.store and self.workflow_id:
                                pending_nodes = (active_nodes - completed_nodes - failed_nodes) | next_active_nodes
                                print(f"DEBUG FAIL_FAST: active={active_nodes}, completed={completed_nodes}, failed={failed_nodes}, next={next_active_nodes}, pending={pending_nodes}")
                                checkpoint = Checkpoint(
                                    workflow_id=self.workflow_id,
                                    status=WorkflowStatus.FAILED,
                                    state_data=state.model_dump(),
                                    completed_nodes=list(completed_nodes),
                                    failed_nodes=list(failed_nodes),
                                    active_nodes=list(pending_nodes),
                                    trace_so_far=trace
                                )
                                self.store.save_checkpoint(checkpoint)
                                self._fire_event("CHECKPOINT_SAVED", workflow_id=self.workflow_id, checkpoint=checkpoint)
                                
                            return GraphExecutionResult(
                                final_state=state,
                                trace=trace,
                                success=False,
                                error=f"Node {node_name} failed: {str(e)}"
                            )
                        elif self.failure_policy == FailurePolicy.CONTINUE_INDEPENDENT:
                            pass
                        elif self.failure_policy == FailurePolicy.SKIP_DEPENDENTS:
                            destinations = []
                            if node_name in self.conditional_edges:
                                if node_name in self.edges:
                                    destinations = [self.edges[node_name]]
                            elif node_name in self.edges:
                                destinations = [self.edges[node_name]]
                                
                            for dest in destinations:
                                if dest != "END":
                                    next_active_nodes.add(dest)
                        
            active_nodes = next_active_nodes
            
            if self.store and self.workflow_id:
                checkpoint = Checkpoint(
                    workflow_id=self.workflow_id,
                    status=WorkflowStatus.RUNNING,
                    state_data=state.model_dump(),
                    completed_nodes=list(completed_nodes),
                    failed_nodes=list(failed_nodes),
                    active_nodes=list(active_nodes),
                    trace_so_far=trace
                )
                self.store.save_checkpoint(checkpoint)
                self._fire_event("CHECKPOINT_SAVED", workflow_id=self.workflow_id, checkpoint=checkpoint)
            
        success = len(failed_nodes) == 0
        
        if self.store and self.workflow_id:
            checkpoint = Checkpoint(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED,
                state_data=state.model_dump(),
                completed_nodes=list(completed_nodes),
                failed_nodes=list(failed_nodes),
                active_nodes=[],
                trace_so_far=trace
            )
            self.store.save_checkpoint(checkpoint)
        
        if success:
            self._fire_event("GRAPH_COMPLETED", state=state)
        else:
            self._fire_event("GRAPH_FAILED", state=state, failed_nodes=list(failed_nodes))
            
        return GraphExecutionResult(
            final_state=state,
            trace=trace,
            success=success,
            error=f"Failed nodes: {failed_nodes}" if not success else None
        )
        
    def _execute_node_with_retry(self, node: GraphNode, state: GraphState) -> tuple[NodeResult, float, int]:
        start_time = time.time()
        attempt = 0
        last_exception = None
        
        while attempt < self.retry_policy.max_retries:
            try:
                if attempt > 0:
                    delay = self.retry_policy.base_delay * (self.retry_policy.backoff_factor ** (attempt - 1))
                    time.sleep(delay)
                    self._fire_event("NODE_RETRY", node=node.name, attempt=attempt)
                    
                result = node.execute(state)
                latency = time.time() - start_time
                return result, latency, attempt + 1
            except Exception as e:
                last_exception = e
                attempt += 1
                logger.warning(f"Node {node.name} attempt {attempt} failed: {e}")
                self._fire_event("NODE_FAILED", node=node.name, error=str(e), attempt=attempt)
                
                if self.store and self.workflow_id:
                    self.store.log_failure(self.workflow_id, FailureRecord(
                        node=node.name,
                        error=str(e),
                        attempt=attempt
                    ))
                
        raise RuntimeError(f"Node {node.name} permanently failed after {self.retry_policy.max_retries} attempts. Last error: {last_exception}")

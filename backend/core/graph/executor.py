import time
import logging
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

from core.graph.contracts import GraphNode, GraphState, GraphExecutionResult, NodeResult, RoutingDecision

logger = logging.getLogger(__name__)

class ExecutionGraph:
    def __init__(self, max_workers: int = 4):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable[[Any], List[str]]] = {}
        self.entry_point: str = ""
        self.max_workers = max_workers

    def add_node(self, node: GraphNode):
        self.nodes[node.name] = node

    def set_entry_point(self, node_name: str):
        self.entry_point = node_name

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edge(self, from_node: str, condition_fn: Callable[[Any], List[str]]):
        self.conditional_edges[from_node] = condition_fn
        
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

    def run(self, state: GraphState) -> GraphExecutionResult:
        self.validate()
        
        trace = []
        active_nodes = [self.entry_point]
        
        while active_nodes:
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
                            "status": "success",
                            "latency": latency,
                            "attempts": attempts,
                            "updates": list(node_result.state_updates.keys())
                        })
                        
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
                            if dest != "END":
                                next_active_nodes.add(dest)
                                
                    except Exception as e:
                        logger.error(f"Node {node_name} failed: {e}")
                        trace.append({
                            "node": node_name,
                            "status": "error",
                            "error": str(e)
                        })
                        return GraphExecutionResult(
                            final_state=state,
                            trace=trace,
                            success=False,
                            error=f"Node {node_name} failed: {str(e)}"
                        )
                        
            active_nodes = list(next_active_nodes)
            
        return GraphExecutionResult(
            final_state=state,
            trace=trace,
            success=True
        )
        
    def _execute_node_with_retry(self, node: GraphNode, state: GraphState, max_retries: int = 3) -> tuple[NodeResult, float, int]:
        start_time = time.time()
        attempt = 0
        last_exception = None
        
        while attempt < max_retries:
            try:
                result = node.execute(state)
                latency = time.time() - start_time
                return result, latency, attempt + 1
            except Exception as e:
                last_exception = e
                attempt += 1
                logger.warning(f"Node {node.name} attempt {attempt} failed: {e}")
                
        raise RuntimeError(f"Node {node.name} permanently failed after {max_retries} attempts. Last error: {last_exception}")

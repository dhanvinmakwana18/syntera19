import time
import uuid
import logging
from typing import Any, Dict
from pydantic import BaseModel, Field

from core.graph.executor import ExecutionGraph
from core.graph.contracts import GraphNode, NodeResult, RoutingDecision, GraphState
from core.durability.sqlite_store import SqliteStore
from core.durability.contracts import RetryPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. Domain State
# ---------------------------------------------------------
class ResearchState(GraphState):
    topic: str
    search_queries: list[str] = Field(default_factory=list)
    raw_data: str = ""
    analysis: str = ""
    draft: str = ""
    verification: str = ""
    is_complete: bool = False
    simulate_crash: bool = False

# ---------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------
class PlannerNode(GraphNode):
    @property
    def name(self) -> str: return "Planner"
    def execute(self, state: ResearchState) -> NodeResult:
        logger.info("[PlannerNode] Generating search plan...")
        time.sleep(0.2) # simulate work
        return NodeResult(state_updates={"search_queries": [f"{state.topic} basics", f"{state.topic} advanced"]})

class RetrievalNode(GraphNode):
    @property
    def name(self) -> str: return "Retrieval"
    def execute(self, state: ResearchState) -> NodeResult:
        logger.info("[RetrievalNode] Fetching sources...")
        time.sleep(0.2)
        return NodeResult(state_updates={"raw_data": f"Data for {state.search_queries}"})

class AnalysisNode(GraphNode):
    @property
    def name(self) -> str: return "Analysis"
    def execute(self, state: ResearchState) -> NodeResult:
        logger.info("[AnalysisNode] Analyzing data...")
        if state.simulate_crash:
            logger.error("[AnalysisNode] CRASHING THE PROCESS INTENTIONALLY!")
            # We raise a special exception that fails the graph so we can test resume
            raise InterruptedError("Simulated Power Failure")
        time.sleep(0.2)
        return NodeResult(state_updates={"analysis": f"Analysis of: {state.raw_data}"})

class SynthesisNode(GraphNode):
    @property
    def name(self) -> str: return "Synthesis"
    def execute(self, state: ResearchState) -> NodeResult:
        logger.info("[SynthesisNode] Writing draft...")
        time.sleep(0.2)
        return NodeResult(state_updates={"draft": f"Draft based on: {state.analysis}"})

class VerificationNode(GraphNode):
    @property
    def name(self) -> str: return "Verification"
    def execute(self, state: ResearchState) -> NodeResult:
        logger.info("[VerificationNode] Verifying draft...")
        time.sleep(0.2)
        return NodeResult(state_updates={"verification": "Verified: OK", "is_complete": True})

# ---------------------------------------------------------
# 3. Demonstration Runner
# ---------------------------------------------------------
def build_graph(workflow_id: str, store: SqliteStore) -> ExecutionGraph:
    graph = ExecutionGraph(workflow_id=workflow_id, store=store, retry_policy=RetryPolicy(max_attempts=1))
    
    graph.add_node(PlannerNode())
    graph.add_node(RetrievalNode())
    graph.add_node(AnalysisNode())
    graph.add_node(SynthesisNode())
    graph.add_node(VerificationNode())
    
    graph.set_entry_point("Planner")
    graph.add_edge("Planner", "Retrieval")
    graph.add_edge("Retrieval", "Analysis")
    graph.add_edge("Analysis", "Synthesis")
    graph.add_edge("Synthesis", "Verification")
    
    return graph

def run_demo():
    print("="*60)
    print("PHASE 11: DURABLE AUTONOMOUS WORKFLOW DEMONSTRATION")
    print("="*60)
    
    store = SqliteStore("demo_durability.db")
    workflow_id = str(uuid.uuid4())
    
    print(f"\n[1] Starting new workflow: {workflow_id}")
    graph = build_graph(workflow_id, store)
    
    # Enable crash simulation
    state = ResearchState(topic="Quantum Computing", simulate_crash=True)
    
    # Run the graph. It will checkpoint after Planner, Retrieval, and then crash at Analysis.
    result = graph.run(state)
    
    print("\n[2] Execution Result After Crash:")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    
    print("\n[3] Restarting the system (Simulating process restart)...")
    time.sleep(1)
    
    # Create an entirely new graph instance (simulating a restarted process)
    store2 = SqliteStore("demo_durability.db")
    graph2 = build_graph(workflow_id, store2)
    
    # Turn off the crash condition in the persisted state before resuming
    # In a real system, the underlying issue would be fixed before resume.
    checkpoint = store2.get_checkpoint(workflow_id)
    checkpoint.state_data["simulate_crash"] = False
    
    # We must explicitly move the failed node back to active to retry it after a manual fix
    if "Analysis" in checkpoint.failed_nodes:
        checkpoint.failed_nodes.remove("Analysis")
        if "Analysis" not in checkpoint.active_nodes:
            checkpoint.active_nodes.append("Analysis")
            
    store2.save_checkpoint(checkpoint)
    
    print("\n[4] Resuming workflow from durable checkpoint...")
    resume_result = graph2.resume(ResearchState)
    
    print("\n[5] Execution Result After Resume:")
    print(f"Success: {resume_result.success}")
    print(f"Final Draft: {resume_result.final_state.draft}")
    print(f"Verified: {resume_result.final_state.verification}")
    
    print("\n[6] Execution Trace:")
    for t in resume_result.trace:
        print(f" - Node: {t['node']}, Status: {t['status']}")
        
    print("\nDemonstration complete.")

if __name__ == "__main__":
    run_demo()

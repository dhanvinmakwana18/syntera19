import pytest
import os
import sqlite3
from typing import Any

from core.graph.contracts import GraphNode, GraphState, NodeResult, FailurePolicy
from core.graph.executor import ExecutionGraph
from core.durability.contracts import RetryPolicy, Checkpoint, WorkflowStatus
from core.durability.sqlite_store import SqliteStore

class MyState(GraphState):
    counter: int = 0
    message: str = ""

class SucceedNode(GraphNode):
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name

    def execute(self, state: Any) -> NodeResult:
        return NodeResult(state_updates={"counter": state.counter + 1, "message": f"{state.message} {self._name}".strip()})

class FlakyNode(GraphNode):
    def __init__(self, name: str, fail_until: int):
        self._name = name
        self.fail_until = fail_until
        self.attempts = 0

    @property
    def name(self) -> str:
        return self._name

    def execute(self, state: Any) -> NodeResult:
        self.attempts += 1
        if self.attempts < self.fail_until:
            raise ValueError(f"Failing on attempt {self.attempts}")
        return NodeResult(state_updates={"counter": state.counter + 1})

class FailNode(GraphNode):
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name

    def execute(self, state: Any) -> NodeResult:
        raise ValueError("Intentional permanent failure")

def test_retry_policy():
    store = SqliteStore()
    retry_policy = RetryPolicy(max_retries=3, base_delay=0.01)
    graph = ExecutionGraph(workflow_id="wf_retry", store=store, retry_policy=retry_policy)
    
    flaky = FlakyNode("flaky1", fail_until=3)
    graph.add_node(flaky)
    graph.set_entry_point("flaky1")
    
    state = MyState()
    result = graph.run(state)
    
    assert result.success
    assert flaky.attempts == 3
    assert result.final_state.counter == 1

def test_checkpointing():
    store = SqliteStore()
    graph = ExecutionGraph(workflow_id="wf_chk", store=store)
    
    n1 = SucceedNode("n1")
    n2 = SucceedNode("n2")
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge("n1", "n2")
    graph.set_entry_point("n1")
    
    state = MyState()
    result = graph.run(state)
    assert result.success
    
    chk = store.get_checkpoint("wf_chk")
    assert chk is not None
    assert chk.status == WorkflowStatus.COMPLETED
    assert set(chk.completed_nodes) == {"n1", "n2"}

def test_resume():
    store = SqliteStore()
    retry_policy = RetryPolicy(max_retries=1) # fail fast
    graph = ExecutionGraph(workflow_id="wf_resume", store=store, failure_policy=FailurePolicy.FAIL_FAST, retry_policy=retry_policy)
    
    n1 = SucceedNode("n1")
    n2 = FailNode("n2")
    n3 = SucceedNode("n3")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    graph.set_entry_point("n1")
    
    state = MyState()
    result = graph.run(state)
    assert not result.success
    
    chk = store.get_checkpoint("wf_resume")
    assert chk is not None
    assert "n1" in chk.completed_nodes
    assert "n2" in chk.failed_nodes
    assert chk.status == WorkflowStatus.FAILED
    
    # We will now fix the graph and resume
    graph2 = ExecutionGraph(workflow_id="wf_resume", store=store)
    graph2.add_node(n1)
    graph2.add_node(SucceedNode("n2")) # Replaced with a succeeding node
    graph2.add_node(n3)
    graph2.add_edge("n1", "n2")
    graph2.add_edge("n2", "n3")
    graph2.set_entry_point("n1")
    
    # In resume, we need to alter the checkpoint so that n2 isn't permanently failed or we change failure policy
    # Actually, a better test for resume is simulating process death (saving checkpoint while running)
    pass

def test_process_termination_resume(tmp_path):
    db_path = str(tmp_path / "test.db")
    
    # First process
    store1 = SqliteStore(db_path)
    graph1 = ExecutionGraph(workflow_id="wf_term", store=store1, failure_policy=FailurePolicy.FAIL_FAST, retry_policy=RetryPolicy(max_retries=1))
    
    n1 = SucceedNode("n1")
    n2 = FailNode("n2") # This will stop execution
    n3 = SucceedNode("n3")
    
    graph1.add_node(n1)
    graph1.add_node(n2)
    graph1.add_node(n3)
    graph1.add_edge("n1", "n2")
    graph1.add_edge("n2", "n3")
    graph1.set_entry_point("n1")
    
    result = graph1.run(MyState())
    assert not result.success
    
    # Now simulate a restart and fix the workflow state in the DB to resume
    # We'll just reset the failed node to active to resume from it
    chk = store1.get_checkpoint("wf_term")
    chk.failed_nodes = []
    chk.active_nodes = ["n2"] # resume from n2
    chk.status = WorkflowStatus.RUNNING
    store1.save_checkpoint(chk)
    
    # Second process
    store2 = SqliteStore(db_path)
    graph2 = ExecutionGraph(workflow_id="wf_term", store=store2)
    
    # We replace n2 with a succeeding node
    graph2.add_node(SucceedNode("n1"))
    graph2.add_node(SucceedNode("n2"))
    graph2.add_node(SucceedNode("n3"))
    graph2.add_edge("n1", "n2")
    graph2.add_edge("n2", "n3")
    graph2.set_entry_point("n1")
    
    result2 = graph2.resume(MyState)
    assert result2.success
    
    # Check that n1 was completed from process 1, n2 and n3 from process 2
    chk2 = store2.get_checkpoint("wf_term")
    assert set(chk2.completed_nodes) == {"n1", "n2", "n3"}
    assert result2.final_state.counter == 3 # n1 (proc1) + n2 (proc2) + n3 (proc2)

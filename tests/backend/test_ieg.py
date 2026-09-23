import pytest
from orchestration.ieg.orchestrator import IEGState, run_ieg
from core.container import build_container

@pytest.fixture
def mock_container():
    container = build_container()
    class FakeLLM:
        def __init__(self):
            self.responses = []
            self.calls = 0
            
        def generate(self, *args, **kwargs):
            from intelligence.contracts import IntelligenceResponse
            response_text = self.responses[self.calls]
            self.calls += 1
            return IntelligenceResponse(content=response_text)
            
    container._instances["intelligence"] = FakeLLM()
    
    class FakePipeline:
        def run(self, *args, **kwargs):
            from core.domain import PipelineResult
            return PipelineResult(query=args[0], candidates=[], trace=[])
            
    class FakeContainer(type(container)):
        def build_pipeline(self, *args, **kwargs):
            return FakePipeline()
            
    container.__class__ = FakeContainer
    return container

def test_ieg_simple_query(mock_container):
    mock_container._instances["intelligence"].responses = [
        '{"subqueries": [{"id": "q1", "query": "test query", "purpose": "test", "dependency": null}]}',
        '{"sufficient": true, "reasoning": "Got everything.", "missing_information": [], "follow_up_queries": []}',
        'Final answer [Source 1]'
    ]
    
    # We mock retrieve by overriding the pipeline run behavior or simply letting it return empty docs, 
    # but the test relies on state.evidence being appended to. 
    # Let's monkeypatch retrieve_parallel instead since it's cleaner for testing orchestration flow.
    import orchestration.ieg.orchestrator as orch
    original_retrieve = orch.retrieve_parallel
    
    def fake_retrieve(state, queries, container=None):
        state.evidence.append({"id": "1", "text": "Mocked evidence", "originating_subquery": "q1"})
        
    orch.retrieve_parallel = fake_retrieve
    try:
        state = run_ieg("What is Syntera?", max_iterations=3, container=mock_container)
    finally:
        orch.retrieve_parallel = original_retrieve
    
    assert state.iteration == 0
    assert len(state.subqueries) == 1
    assert state.evaluations[0].sufficient is True
    assert state.final_answer == "Final answer [Source 1]"
    assert "Evidence Evaluator deemed context sufficient" in state.termination_reason

def test_ieg_insufficient_then_sufficient(mock_container):
    mock_container._instances["intelligence"].responses = [
        '{"subqueries": [{"id": "q1", "query": "test query", "purpose": "test"}]}',
        '{"sufficient": false, "reasoning": "Missing detail", "missing_information": ["detail"], "follow_up_queries": [{"id": "q2", "query": "detail", "purpose": "find detail"}]}',
        '{"sufficient": true, "reasoning": "Got detail.", "missing_information": [], "follow_up_queries": []}',
        'Final answer with detail [Source 2]'
    ]
    
    import orchestration.ieg.orchestrator as orch
    original_retrieve = orch.retrieve_parallel
    
    def fake_retrieve(state, queries, container=None):
        for q in queries:
            state.evidence.append({"id": q.id, "text": f"Mocked evidence for {q.id}", "originating_subquery": q.id})
            
    orch.retrieve_parallel = fake_retrieve
    try:
        state = run_ieg("What is Syntera's detail?", max_iterations=3, container=mock_container)
    finally:
        orch.retrieve_parallel = original_retrieve
    
    assert state.iteration == 1
    assert len(state.subqueries) == 2  # q1 and iter_0_q2
    assert state.evaluations[0].sufficient is False
    assert state.evaluations[1].sufficient is True
    assert "Final answer" in state.final_answer

def test_ieg_max_iterations(mock_container):
    mock_container._instances["intelligence"].responses = [
        '{"subqueries": [{"id": "q1", "query": "test query", "purpose": "test"}]}',
        '{"sufficient": false, "reasoning": "Missing detail", "missing_information": ["detail"], "follow_up_queries": [{"id": "q2", "query": "detail", "purpose": "find detail"}]}',
        '{"sufficient": false, "reasoning": "Still missing", "missing_information": ["more"], "follow_up_queries": [{"id": "q3", "query": "more", "purpose": "more"}]}',
        'Final answer anyway'
    ]
    
    import orchestration.ieg.orchestrator as orch
    original_retrieve = orch.retrieve_parallel
    
    def fake_retrieve(state, queries, container=None):
        pass
        
    orch.retrieve_parallel = fake_retrieve
    try:
        state = run_ieg("Impossible question?", max_iterations=2, container=mock_container)
    finally:
        orch.retrieve_parallel = original_retrieve
    
    assert state.iteration == 2
    assert state.termination_reason == "Maximum iterations reached."

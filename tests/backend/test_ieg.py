import pytest
from unittest.mock import patch, MagicMock
from orchestration.ieg.orchestrator import IEGState, run_ieg
from orchestration.ieg.schemas import EvidenceEvaluation, SubQuery

@pytest.fixture
def mock_settings(monkeypatch):
    from core.config import settings
    settings.LLM_PROVIDER = "ollama"
    settings.LLM_BASE_URL = ""
    settings.ASTRA_API_KEY = ""

@patch("orchestration.ieg.orchestrator.retrieve_parallel")
@patch("orchestration.ieg.orchestrator.llm_provider.generate")
def test_ieg_simple_query(mock_generate, mock_retrieve, mock_settings):
    # Mock decomposition
    mock_generate.side_effect = [
        # Decomposition response
        '{"subqueries": [{"id": "q1", "query": "test query", "purpose": "test", "dependency": null}]}',
        # Evaluator response (Sufficient)
        '{"sufficient": true, "reasoning": "Got everything.", "missing_information": [], "follow_up_queries": []}',
        # Synthesis response
        'Final answer [Source 1]'
    ]
    
    # Mock retrieval to append some evidence
    def fake_retrieve(state, queries):
        state.evidence.append({"id": "1", "text": "Mocked evidence", "originating_subquery": "q1"})
        
    mock_retrieve.side_effect = fake_retrieve
    
    state = run_ieg("What is Syntera?", max_iterations=3)
    
    assert state.iteration == 0
    assert len(state.subqueries) == 1
    assert state.evaluations[0].sufficient is True
    assert state.final_answer == "Final answer [Source 1]"
    assert "Evidence Evaluator deemed context sufficient" in state.termination_reason

@patch("orchestration.ieg.orchestrator.retrieve_parallel")
@patch("orchestration.ieg.orchestrator.llm_provider.generate")
def test_ieg_insufficient_then_sufficient(mock_generate, mock_retrieve, mock_settings):
    # Mock decomposition
    mock_generate.side_effect = [
        # Decomposition response (Iteration 0)
        '{"subqueries": [{"id": "q1", "query": "test query", "purpose": "test"}]}',
        # Evaluator response (Iteration 0: Insufficient)
        '{"sufficient": false, "reasoning": "Missing detail", "missing_information": ["detail"], "follow_up_queries": [{"id": "q2", "query": "detail", "purpose": "find detail"}]}',
        # Evaluator response (Iteration 1: Sufficient)
        '{"sufficient": true, "reasoning": "Got detail.", "missing_information": [], "follow_up_queries": []}',
        # Synthesis response
        'Final answer with detail [Source 2]'
    ]
    
    # Mock retrieval to append evidence sequentially
    def fake_retrieve(state, queries):
        for q in queries:
            state.evidence.append({"id": q.id, "text": f"Mocked evidence for {q.id}", "originating_subquery": q.id})
            
    mock_retrieve.side_effect = fake_retrieve
    
    state = run_ieg("What is Syntera's detail?", max_iterations=3)
    
    assert state.iteration == 1
    assert len(state.subqueries) == 2  # q1 and iter_0_q2
    assert state.evaluations[0].sufficient is False
    assert state.evaluations[1].sufficient is True
    assert "Final answer" in state.final_answer

@patch("orchestration.ieg.orchestrator.retrieve_parallel")
@patch("orchestration.ieg.orchestrator.llm_provider.generate")
def test_ieg_max_iterations(mock_generate, mock_retrieve, mock_settings):
    # Mock decomposition
    mock_generate.side_effect = [
        # Decomposition response (Iteration 0)
        '{"subqueries": [{"id": "q1", "query": "test query", "purpose": "test"}]}',
        # Evaluator response (Iteration 0: Insufficient)
        '{"sufficient": false, "reasoning": "Missing detail", "missing_information": ["detail"], "follow_up_queries": [{"id": "q2", "query": "detail", "purpose": "find detail"}]}',
        # Evaluator response (Iteration 1: Insufficient)
        '{"sufficient": false, "reasoning": "Still missing", "missing_information": ["more"], "follow_up_queries": [{"id": "q3", "query": "more", "purpose": "more"}]}',
        # Synthesis response
        'Final answer anyway'
    ]
    
    def fake_retrieve(state, queries):
        pass
        
    mock_retrieve.side_effect = fake_retrieve
    
    # Restrict max_iterations to 2
    state = run_ieg("Impossible question?", max_iterations=2)
    
    assert state.iteration == 2
    assert state.termination_reason == "Maximum iterations reached."

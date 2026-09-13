import pytest
from evolution.contracts import SelectionPolicy, SelectionRule, SelectionObjective
from evolution.selector import EvolutionSelector
from intelligence.evaluation import EvalRunResult, MetricResult

def test_evolution_selector_maximize_success():
    policy = SelectionPolicy(
        rules=[SelectionRule(objective=SelectionObjective.MAXIMIZE_SUCCESS, weight=1.0)],
        strict_regression_rejection=True
    )
    selector = EvolutionSelector(policy)
    
    parent = EvalRunResult(
        dataset_name="test", system_name="v1", total_samples=10, successful_samples=5,
        metrics=[MetricResult(name="success_rate", value=0.5)]
    )
    
    # Candidate 1 is worse (regression)
    c1 = EvalRunResult(
        dataset_name="test", system_name="c1", total_samples=10, successful_samples=4,
        metrics=[MetricResult(name="success_rate", value=0.4)]
    )
    
    # Candidate 2 is better
    c2 = EvalRunResult(
        dataset_name="test", system_name="c2", total_samples=10, successful_samples=8,
        metrics=[MetricResult(name="success_rate", value=0.8)]
    )
    
    # Should reject regression
    assert selector.select_best(parent, [c1]) is None
    
    # Should promote c2
    best = selector.select_best(parent, [c1, c2])
    assert best is not None
    assert best.system_name == "c2"

def test_evolution_selector_tiebreaker():
    policy = SelectionPolicy(
        rules=[
            SelectionRule(objective=SelectionObjective.MAXIMIZE_SUCCESS, weight=10.0),
            SelectionRule(objective=SelectionObjective.MINIMIZE_LATENCY, weight=1.0)
        ]
    )
    selector = EvolutionSelector(policy)
    
    parent = EvalRunResult(
        dataset_name="test", system_name="v1", total_samples=10, successful_samples=5,
        total_latency_ms=1000,
        metrics=[MetricResult(name="success_rate", value=0.5), MetricResult(name="mean_latency_ms", value=100)]
    )
    
    # Candidate 1: Same success, higher latency
    c1 = EvalRunResult(
        dataset_name="test", system_name="c1", total_samples=10, successful_samples=5,
        total_latency_ms=2000,
        metrics=[MetricResult(name="success_rate", value=0.5), MetricResult(name="mean_latency_ms", value=200)]
    )
    
    # Candidate 2: Same success, lower latency
    c2 = EvalRunResult(
        dataset_name="test", system_name="c2", total_samples=10, successful_samples=5,
        total_latency_ms=500,
        metrics=[MetricResult(name="success_rate", value=0.5), MetricResult(name="mean_latency_ms", value=50)]
    )
    
    # c1 should be rejected (worse than parent on secondary)
    assert selector.select_best(parent, [c1]) is None
    
    # c2 should win over parent (same success, better latency)
    best = selector.select_best(parent, [c1, c2])
    assert best is not None
    assert best.system_name == "c2"

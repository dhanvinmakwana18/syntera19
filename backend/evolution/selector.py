"""
Syntera Evolution Engine - Selector
"""
from typing import List, Optional
from intelligence.evaluation import EvalRunResult
from evolution.contracts import SelectionPolicy, SelectionObjective, EvolutionCandidate

class EvolutionSelector:
    def __init__(self, policy: SelectionPolicy):
        self.policy = policy
        
    def select_best(
        self, 
        parent_result: EvalRunResult, 
        candidate_results: List[EvalRunResult]
    ) -> Optional[EvalRunResult]:
        """
        Returns the best candidate EvalRunResult if it beats the parent, otherwise None.
        """
        if not candidate_results:
            return None
            
        all_results = [parent_result] + candidate_results
        
        # Calculate a score for each result based on the policy rules
        def score_result(res: EvalRunResult) -> float:
            score = 0.0
            
            # Map metric names to values for easy lookup
            metrics = {m.name: m.value for m in res.metrics}
            # Add implicit metrics
            metrics["success_rate"] = metrics.get("success_rate", res.successful_samples / res.total_samples if res.total_samples else 0)
            metrics["mean_latency_ms"] = metrics.get("mean_latency_ms", res.total_latency_ms / res.total_samples if res.total_samples else 0)
            
            for rule in self.policy.rules:
                if rule.objective == SelectionObjective.MAXIMIZE_SUCCESS:
                    val = metrics.get("success_rate", 0.0)
                    if rule.threshold is not None and val < rule.threshold:
                        return -999999.0 # Fail threshold
                    score += val * rule.weight
                    
                elif rule.objective == SelectionObjective.MINIMIZE_LATENCY:
                    val = metrics.get("mean_latency_ms", 10000.0)
                    if rule.threshold is not None and val > rule.threshold:
                        return -999999.0
                    # Inverse for minimization (lower is better, so negate it)
                    score -= (val / 1000.0) * rule.weight 
                    
                elif rule.objective == SelectionObjective.MINIMIZE_COST:
                    val = metrics.get("cost", 0.0)
                    score -= val * rule.weight
                    
                elif rule.objective == SelectionObjective.MAXIMIZE_EXACT_MATCH:
                    val = metrics.get("exact_match", 0.0)
                    score += val * rule.weight
                    
            return score
            
        # Score and sort
        scored_results = [(score_result(res), res) for res in all_results]
        scored_results.sort(key=lambda x: x[0], reverse=True) # Highest score first
        
        best_score, best_res = scored_results[0]
        parent_score = score_result(parent_result)
        
        # Check strict regression
        if self.policy.strict_regression_rejection:
            # If the best candidate's primary metric (first rule) is worse than parent, reject
            primary_rule = self.policy.rules[0] if self.policy.rules else None
            if primary_rule:
                metrics_best = {m.name: m.value for m in best_res.metrics}
                metrics_best["success_rate"] = metrics_best.get("success_rate", best_res.successful_samples / best_res.total_samples if best_res.total_samples else 0)
                metrics_parent = {m.name: m.value for m in parent_result.metrics}
                metrics_parent["success_rate"] = metrics_parent.get("success_rate", parent_result.successful_samples / parent_result.total_samples if parent_result.total_samples else 0)
                
                if primary_rule.objective == SelectionObjective.MAXIMIZE_SUCCESS:
                    if metrics_best.get("success_rate", 0) < metrics_parent.get("success_rate", 0):
                        return None
        
        # If the best is the parent or best score isn't strictly better, return None (reject)
        if best_res.run_id == parent_result.run_id or best_score <= parent_score:
            return None
            
        return best_res

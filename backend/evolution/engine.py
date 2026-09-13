"""
Syntera Evolution Engine - Core Engine
"""
import copy
from typing import List, Callable
from intelligence.core import IntelligenceCore
from core.creation.builder import AISystemBuilder
from intelligence.evaluation import EvalDataset, EvaluationRunner, EvalRunResult
from core.creation.domain import GeneratedAISystem
from evolution.contracts import EvolutionEvent, SelectionPolicy, CandidateStatus
from evolution.diagnostics import DiagnosticEngine
from evolution.generator import CandidateGenerator
from evolution.selector import EvolutionSelector
from core.graph.contracts import GraphState
from core.domain import Query

class EvolutionEngine:
    def __init__(
        self,
        intelligence: IntelligenceCore,
        builder: AISystemBuilder,
        eval_runner: EvaluationRunner,
        policy: SelectionPolicy
    ):
        self.intelligence = intelligence
        self.builder = builder
        self.eval_runner = eval_runner
        
        self.diagnostics = DiagnosticEngine(intelligence)
        self.generator = CandidateGenerator(intelligence)
        self.selector = EvolutionSelector(policy)
        
        self.event_hooks: List[Callable[[EvolutionEvent, dict], None]] = []
        self.history: List[dict] = []
        
    def add_hook(self, hook: Callable[[EvolutionEvent, dict], None]):
        self.event_hooks.append(hook)
        
    def _emit(self, event: EvolutionEvent, data: dict):
        self.history.append({"event": event.value, "data": data})
        for hook in self.event_hooks:
            try:
                hook(event, data)
            except Exception:
                pass
                
    def _run_eval(self, system: GeneratedAISystem, dataset: EvalDataset) -> EvalRunResult:
        def evaluate_system(input_text: str) -> str:
            state = GraphState(
                query=Query(text=input_text),
                documents=[], context=None, generation_result=None,
                verification_result=None, plan=None, observations=[],
                current_task_idx=0, iteration=0, max_iterations=3,
                final_answer=None, error=None, metadata={}
            )
            res = system.execution_graph.run(state)
            if res.final_state.error:
                raise RuntimeError(res.final_state.error)
            return res.final_state.final_answer or ""
            
        return self.eval_runner.run(dataset, evaluate_system, system_name=system.identity.name)

    def evolve(
        self, 
        system: GeneratedAISystem, 
        dataset: EvalDataset, 
        max_iterations: int = 1,
        num_candidates: int = 1
    ) -> GeneratedAISystem:
        """
        Runs the evolution lifecycle on a system.
        """
        current_system = system
        
        for iteration in range(1, max_iterations + 1):
            sys_id = current_system.identity.id if hasattr(current_system.identity, 'id') else current_system.identity.name
            self._emit(EvolutionEvent.START, {"iteration": iteration, "system_id": sys_id})
            
            # 1. Evaluate baseline if not present
            baseline_result = current_system.metadata.get("eval_result")
            if not baseline_result:
                baseline_result = self._run_eval(current_system, dataset)
                current_system.metadata["eval_result"] = baseline_result
                
            # If baseline is perfect, stop
            if baseline_result.successful_samples == baseline_result.total_samples:
                self._emit(EvolutionEvent.PROMOTED, {"reason": "Perfect score achieved", "system_id": sys_id})
                break
                
            # 2. Diagnose
            diagnosis = self.diagnostics.diagnose(current_system.specification, baseline_result)
            self._emit(EvolutionEvent.DIAGNOSIS, {"weaknesses": diagnosis.weaknesses, "suggested": diagnosis.suggested_improvements})
            
            # 3. Generate Candidates
            candidates = self.generator.generate_candidates(sys_id, current_system.specification, diagnosis, num_candidates)
            self._emit(EvolutionEvent.CANDIDATE_GENERATED, {"count": len(candidates)})
            
            if not candidates:
                self._emit(EvolutionEvent.ERROR, {"reason": "No candidates generated"})
                break
                
            candidate_results = []
            
            # 4. Build and Evaluate Candidates
            for candidate in candidates:
                try:
                    candidate.built_system = self.builder.build(candidate.proposed_spec)
                    candidate.status = CandidateStatus.BUILT
                    self._emit(EvolutionEvent.CANDIDATE_BUILT, {"candidate_id": candidate.id})
                    
                    # Evaluate
                    candidate.eval_result = self._run_eval(candidate.built_system, dataset)
                    candidate.built_system.metadata["eval_result"] = candidate.eval_result
                    candidate.status = CandidateStatus.EVALUATED
                    self._emit(EvolutionEvent.CANDIDATE_EVALUATED, {"candidate_id": candidate.id, "metrics": [m.dict() for m in candidate.eval_result.metrics]})
                    
                    candidate_results.append(candidate.eval_result)
                except Exception as e:
                    candidate.status = CandidateStatus.ERROR
                    candidate.error_message = str(e)
                    self._emit(EvolutionEvent.ERROR, {"candidate_id": candidate.id, "error": str(e)})
                    
            # 5. Compare & Promote
            best_eval = self.selector.select_best(baseline_result, candidate_results)
            self._emit(EvolutionEvent.COMPARISON, {"baseline_run_id": baseline_result.run_id, "best_candidate_run_id": best_eval.run_id if best_eval else None})
            
            if best_eval:
                # Find the winning system
                winner = next(c for c in candidates if c.eval_result and c.eval_result.run_id == best_eval.run_id)
                winner.status = CandidateStatus.PROMOTED
                current_system = winner.built_system
                self._emit(EvolutionEvent.PROMOTED, {"promoted_candidate_id": winner.id})
            else:
                self._emit(EvolutionEvent.REJECTED, {"reason": "No candidate outperformed the baseline (regression detected)"})
                # Rollback/stop since we couldn't improve
                break
                
        return current_system

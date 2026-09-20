"""
Syntera Evolution Engine - Diagnostics
"""
from intelligence.core import IntelligenceCore
from core.creation.domain import AISystemSpecification
from intelligence.evaluation import EvalRunResult
from evolution.contracts import EvolutionDiagnosis
import json

class DiagnosticEngine:
    def __init__(self, intelligence: IntelligenceCore):
        self.intelligence = intelligence
        
    def diagnose(self, spec: AISystemSpecification, eval_result: EvalRunResult) -> EvolutionDiagnosis:
        """
        Analyzes the system specification against the evaluation results to pinpoint weaknesses.
        """
        # Build prompt context
        prompt = (
            "You are an AI Architect diagnosing a system's evaluation performance.\n\n"
            f"System Specification:\n{spec.model_dump_json(indent=2)}\n\n"
            "Evaluation Metrics:\n"
        )
        
        for metric in eval_result.metrics:
            prompt += f"- {metric.name}: {metric.value}\n"
            
        failed_predictions = [p for p in eval_result.predictions if not p.success]
        prompt += f"\nTotal failed samples: {len(failed_predictions)}\n"
        
        # Add sample of failures if they exist
        if failed_predictions:
            prompt += "\nSample Failures:\n"
            for p in failed_predictions[:3]:
                prompt += f"Input: {p.sample_id} | Output: {p.output} | Error: {p.error}\n"
                
        prompt += (
            "\nAnalyze the weaknesses. Why did it fail? What capabilities, knowledge, or tools are missing? "
            "Suggest actionable changes to the AISystemSpecification (e.g., adding capabilities like 'planning' or 'verification', "
            "or changing the system prompt)."
        )
        
        result = self.intelligence.structured_generate(
            prompt=prompt,
            schema=EvolutionDiagnosis,
            system_prompt="You are an expert AI diagnostician. Output precise, actionable technical weaknesses and improvements."
        )
        
        if not result.success or not result.data:
            # Fallback diagnosis
            return EvolutionDiagnosis(
                weaknesses=["Diagnostic generation failed or timed out.", str(result.error)],
                suggested_improvements=["Review configuration manually or try running diagnostics again."]
            )
            
        return result.data

"""
Syntera Evolution Engine - Candidate Generator
"""
import copy
import uuid
from typing import List
from intelligence.core import IntelligenceCore
from core.creation.domain import AISystemSpecification
from evolution.contracts import EvolutionDiagnosis, CandidateSpec, EvolutionCandidate

class CandidateGenerator:
    def __init__(self, intelligence: IntelligenceCore):
        self.intelligence = intelligence
        
    def generate_candidates(
        self, 
        parent_id: str, 
        spec: AISystemSpecification, 
        diagnosis: EvolutionDiagnosis, 
        num_candidates: int = 1
    ) -> List[EvolutionCandidate]:
        """Generates improved candidate specifications based on the diagnosis."""
        
        prompt = (
            "You are an AI Architect tasked with improving an existing AISystemSpecification.\n"
            f"Original Spec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Diagnosed Weaknesses:\n{diagnosis.weaknesses}\n\n"
            f"Suggested Improvements:\n{diagnosis.suggested_improvements}\n\n"
            "Generate an improved version of the AISystemSpecification that addresses these weaknesses. "
            "You may add capabilities (e.g., 'planning', 'verification', 'tool_use', 'retrieval'), modify the purpose, "
            "or adjust the routing logic. Ensure you provide a rationale for the changes."
        )
        
        candidates = []
        for i in range(num_candidates):
            # We generate them sequentially here to avoid parallel requests blocking on local CPU models
            result = self.intelligence.structured_generate(
                prompt=prompt,
                schema=CandidateSpec,
                system_prompt="Output exactly one CandidateSpec with a 'rationale' and a fully valid 'proposed_spec'.",
                temperature=0.7 # Add variance for multiple candidates
            )
            
            if result.success and result.data:
                candidate = EvolutionCandidate(
                    id=str(uuid.uuid4())[:8],
                    parent_system_id=parent_id,
                    proposed_spec=result.data.proposed_spec,
                    rationale=result.data.rationale
                )
                candidates.append(candidate)
                
        return candidates

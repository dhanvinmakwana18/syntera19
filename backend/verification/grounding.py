import re
from core.contracts import BaseVerifier
from core.domain import Query, GenerationContext, GenerationResult, VerificationResult

def validate_citations(response: str, allowed_sources: list) -> str:
    valid_ids = {str(s["id"]) for s in allowed_sources}
    cited_ids = set(re.findall(r'\[Source (\d+)\]', response))
    invalid_ids = cited_ids - valid_ids
    if invalid_ids:
        warning = f"\n\n[SYSTEM WARNING: The model cited sources that do not exist: {', '.join(invalid_ids)}]"
        return response + warning
    return response

def evaluate_support(claim: str, evidence: str) -> bool:
    from providers.llm import llm_provider
    prompt = f"Evidence:\n{evidence}\n\nClaim:\n{claim}\n\nBased ONLY on the evidence above, is the claim fully supported? Answer strictly with YES or NO."
    try:
        system_prompt = "You are a strict logical validator. Evaluate entailment and output only YES or NO."
        res = llm_provider.generate(prompt=prompt, system_prompt=system_prompt).strip().upper()
        return "YES" in res
    except Exception:
        return False

class CitationVerifier(BaseVerifier):
    def verify(self, query: Query, context: GenerationContext, response: GenerationResult, **kwargs) -> VerificationResult:
        valid_ids = {str(s["id"]) for s in context.sources}
        cited_ids = set(re.findall(r'\[Source (\d+)\]', response.answer))
        
        invalid_ids = cited_ids - valid_ids
        
        if invalid_ids:
            return VerificationResult(
                passed=False, 
                reason=f"Model cited non-existent sources: {', '.join(invalid_ids)}",
                metrics={"invalid_citations": len(invalid_ids)}
            )
            
        return VerificationResult(
            passed=True, 
            reason="All citations are valid.",
            metrics={"valid_citations": len(cited_ids)}
        )


def register(registry):
    registry.register_verifier("citation", lambda **kwargs: CitationVerifier())

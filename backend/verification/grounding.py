import re

def validate_citations(response: str, allowed_sources: list) -> str:
    """
    Validates that any [Source X] cited in the response actually exists
    in the allowed_sources list. If a hallucinated source is found,
    it can either be stripped, or the entire response can be flagged.
    For this implementation, we will append a system warning if hallucinated
    sources are used.
    
    allowed_sources: List of source dicts returned by assemble_context.
    """
    valid_ids = {str(s["id"]) for s in allowed_sources}
    
    # Find all [Source X] in the response
    cited_ids = set(re.findall(r'\[Source (\d+)\]', response))
    
    invalid_ids = cited_ids - valid_ids
    
    if invalid_ids:
        warning = f"\n\n[SYSTEM WARNING: The model cited sources that do not exist: {', '.join(invalid_ids)}]"
        return response + warning
        
    return response

def evaluate_support(claim: str, evidence: str) -> bool:
    """
    Evaluates whether the given claim is semantically supported by the evidence.
    Uses the configured LLM provider to perform the entailment check.
    """
    from providers.llm import llm_provider
    prompt = f"Evidence:\n{evidence}\n\nClaim:\n{claim}\n\nBased ONLY on the evidence above, is the claim fully supported? Answer strictly with YES or NO."
    try:
        system_prompt = "You are a strict logical validator. Evaluate entailment and output only YES or NO."
        res = llm_provider.generate(prompt=prompt, system_prompt=system_prompt).strip().upper()
        return "YES" in res
    except Exception:
        return False

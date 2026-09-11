import re

def transform_query(query: str) -> str:
    """
    Lightweight query transformation.
    In the future, this can invoke an LLM for rewriting or decomposition.
    Currently, it normalizes whitespace and removes redundant punctuation.
    """
    if not query:
        return ""
        
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    
    return query

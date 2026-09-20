def route_query(query: str, container=None) -> str:
    """
    Decides the route based on the query. 
    Routes: DIRECT, RAG, MULTI_MODAL, AGENTIC
    """
    prompt = f"""You are a routing agent. Classify the user's query into exactly one of these categories:
- DIRECT: General knowledge, greetings, or simple tasks.
- RAG: Questions asking about uploaded documents, specific knowledge, or requiring search.
- MULTI_MODAL: Questions about images or diagrams.
- AGENTIC: Complex multi-step reasoning, comparison across multiple documents.

Query: '{query}'

Respond with ONLY the category name."""
    
    llm = container.get_llm()
    response = llm.generate(prompt=prompt, system_prompt="You are a strict routing system.").strip().upper()
    
    valid_routes = ["DIRECT", "RAG", "MULTI_MODAL", "AGENTIC", "IEG"]
    for route in valid_routes:
        if route in response:
            return route
    return "DIRECT"  # Default fallback

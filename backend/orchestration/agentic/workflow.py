import json
from typing import List, Dict, Any

class AgentState:
    def __init__(self, query: str):
        self.query = query
        self.plan: str = ""
        self.tools_selected: List[str] = []
        self.observations: List[Dict[str, Any]] = []
        self.response: str = ""
        self.status: str = "ANALYZING" # ANALYZING, PLANNING, EXECUTING, VERIFYING, RESPONDING, ERROR
        self.iteration: int = 0
        self.max_iterations: int = 3
        self.trace: List[Dict[str, str]] = []

    def add_trace(self, step: str, action: str):
        self.trace.append({"step": step, "action": action})
        
def execute_agent(query: str, container=None) -> AgentState:
    state = AgentState(query)
    state.add_trace("Init", f"Starting agent workflow for query: '{query}'")
    llm = container.get_intelligence()
    
    # 1. ANALYZE
    state.status = "ANALYZING"
    analysis_prompt = f"Analyze this query and decide if we need RAG (retrieval), VISION (image), or DIRECT answer. Query: {query}\nRespond with just RAG, VISION, or DIRECT."
    intent = llm.generate(prompt=analysis_prompt, system_prompt="You are a query analyzer.").content.strip().upper()
    state.add_trace("Analyze", f"Intent classified as {intent}")
    
    # 2. PLAN & SELECT TOOLS
    if "RAG" in intent:
        state.tools_selected.append("retrieve_documents")
    elif "VISION" in intent:
        state.tools_selected.append("analyze_image")
        
    state.plan = f"Will execute tools: {', '.join(state.tools_selected)}"
    state.add_trace("Plan", state.plan)
    
    # 3. EXECUTE
    state.status = "EXECUTING"
    context = ""
    sources = []
    
    if "retrieve_documents" in state.tools_selected:
        try:
            from core.domain import Query
            from retrieval.assembler import ContextBuilder
            pipeline = container.build_pipeline()
            result = pipeline.run(Query(text=query), limit=5)
            context, sources = ContextBuilder().build(result.candidates)
            state.add_trace("Execute", f"Retrieved {len(sources)} documents")
            state.observations.append({"tool": "retrieve_documents", "result": f"Found {len(sources)} documents"})
        except Exception as e:
            state.add_trace("Error", f"Retrieval failed: {str(e)}")
            
    # 4. RESPOND
    state.status = "RESPONDING"
    
    if "RAG" in intent and not context:
        state.response = "I cannot find sufficient evidence in the retrieved documents to answer your question."
        state.add_trace("Respond", "Insufficient evidence")
    else:
        system_prompt = (
            "You are Syntera. Use the provided context to answer the user query.\n"
            "If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n"
            "Always cite your sources using [Source X] notation. NEVER fabricate a source."
        )
        prompt = f"Context:\n{context}\n\nQuery: {query}" if context else f"Query: {query}"
        
        try:
            raw_response = llm.generate(prompt=prompt, system_prompt=system_prompt).content
            if "RAG" in intent and sources:
                from verification.grounding import validate_citations
                state.response = validate_citations(raw_response, sources)
            else:
                state.response = raw_response
            state.add_trace("Respond", "Generated response via LLM")
        except Exception as e:
            state.response = f"Error generating response: {e}"
            state.add_trace("Error", str(e))
        
    # Return structure
    return state, sources

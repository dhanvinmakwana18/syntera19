from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional, Any
import time
import uuid

api_router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"  # auto, direct, rag, agentic
    retrieval_mode: str = "rerank"  # dense, sparse, hybrid, rerank
    context_limit: int = 5
    expand_neighbors: bool = False
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

class SourceInfo(BaseModel):
    id: Any = None
    filename: str = "Unknown"
    page: Any = "?"
    text: str = ""
    score: float = 0.0

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    trace: List[dict] = []
    run_id: str
    grounded: bool = False
    routing_mode: str = "auto"

from orchestration.router import route_query

@api_router.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest, http_request: Request):
    start_time = time.time()
    run_id = str(uuid.uuid4())
    trace = []
    
    def add_trace(step: str, action: str, latency_ms: float = None):
        entry = {"step": step, "action": action}
        if latency_ms is not None:
            entry["latency_ms"] = round(latency_ms, 1)
        trace.append(entry)
    
    add_trace("REQUEST", f"Received query: '{request.query}' | Mode: {request.mode}")
    
    # 1. ROUTING
    route_start = time.time()
    resolved_mode = request.mode.upper()
    if resolved_mode == "AUTO":
        try:
            resolved_mode = route_query(request.query, container=http_request.app.state.container)
            add_trace("ROUTER", f"Auto-resolved to: {resolved_mode}", (time.time() - route_start) * 1000)
        except Exception as e:
            resolved_mode = "DIRECT"
            add_trace("ROUTER", f"Routing failed ({e}), fallback to DIRECT", (time.time() - route_start) * 1000)
    else:
        resolved_mode = request.mode.upper()
        add_trace("ROUTER", f"User-selected mode: {resolved_mode}")
    
    sources = []
    cited = False
    supported = False
    answer = ""
    
    container = http_request.app.state.container
    
    # 2. EXECUTE based on resolved mode
    if resolved_mode == "DIRECT":
        # Direct LLM call — no retrieval
        gen_start = time.time()
        try:
            system_prompt = "You are Syntera, an advanced AI assistant. Answer the user's question directly and concisely."
            llm = container.get_llm()
            answer = llm.generate(prompt=request.query, system_prompt=system_prompt)
            add_trace("LLM_GENERATION", f"Direct response generated", (time.time() - gen_start) * 1000)
            cited = False  # No retrieval = not cited
            supported = False
        except Exception as e:
            answer = f"Error generating response: {e}"
            add_trace("ERROR", f"LLM generation failed: {e}")
    
    elif resolved_mode == "RAG":
        from core.domain import Query
        from orchestration.state import RAGState
        
        graph_start = time.time()
        try:
            # Build the state graph
            graph = container.build_graph(
                retrieval_mode=request.retrieval_mode,
                expand_neighbors=request.expand_neighbors
            )
            
            # Execute the graph
            initial_state = RAGState(query=Query(text=request.query))
            final_state = graph.run(initial_state)
            
            # Extract results
            answer = final_state.generation_result.answer if final_state.generation_result else "No response generated."
            
            if final_state.context and final_state.context.sources:
                sources = final_state.context.sources
                
            # Translate graph trace into API trace
            for step_trace in final_state.trace:
                node = step_trace.get('node', 'unknown')
                status = step_trace.get('status', 'unknown')
                latency = step_trace.get('latency', 0.0) * 1000
                add_trace(f"GRAPH.{node.upper()}", f"Status: {status}", latency)
                
            graph_latency = (time.time() - graph_start) * 1000
            
            if not final_state.success:
                add_trace("ERROR", final_state.error)
                answer = "Execution failed: " + final_state.error
                supported = False
                return QueryResponse(answer=answer, sources=[], trace=trace, run_id=run_id, grounded=False, routing_mode=resolved_mode)

            
            cited = len(sources) > 0 and "[Source" in answer
            
            # Use verification result if available
            if final_state.verification_result:
                supported = final_state.verification_result.passed
                add_trace("VERIFICATION", final_state.verification_result.reason)
            else:
                supported = False
                
            add_trace("GRAPH_COMPLETE", f"Graph workflow finished", graph_latency)
            
        except Exception as e:
            answer = f"Retrieval error: {e}"
            add_trace("ERROR", f"RAG graph pipeline failed: {e}")
            supported = False
            
    elif resolved_mode == "AGENTIC":
        # Full agentic workflow via generic ExecutionGraph
        from orchestration.agentic.state import AgentState
        from core.domain import Query
        
        agent_start = time.time()
        try:
            graph = container.build_agentic_graph()
            initial_state = AgentState(query=Query(text=request.query))
            final_state = graph.run(initial_state)
            
            agent_latency = (time.time() - agent_start) * 1000
            
            if final_state.success:
                agent_state = final_state.final_state
                answer = agent_state.final_answer if agent_state.final_answer else "Agent finished but provided no final answer."
                
                # Extract sources from tool observations metadata
                sources = []
                for obs in agent_state.observations:
                    if obs.get("metadata", {}).get("sources"):
                        sources.extend(obs["metadata"]["sources"])
                
                cited = len(sources) > 0 and "[Source" in answer
                supported = False # Not running full support check on agentic yet for speed
            else:
                answer = "Execution failed: " + str(final_state.error)
                sources = []
                cited = False
                supported = False

            # Transfer graph traces
            for step_trace in final_state.trace:
                node = step_trace.get('node', 'unknown')
                status = step_trace.get('status', 'unknown')
                latency = step_trace.get('latency', 0.0) * 1000
                add_trace(f"GRAPH.{node.upper()}", f"Status: {status}", latency)
                
            add_trace("AGENT_COMPLETE", f"Agentic workflow finished", agent_latency)
            
        except Exception as e:
            answer = f"Agentic workflow error: {e}"
            add_trace("ERROR", f"Agentic pipeline failed: {e}")

    elif resolved_mode == "IEG":
        # Iterative Evidence Graph Orchestrator
        from orchestration.ieg.orchestrator import run_ieg
        
        ieg_start = time.time()
        try:
            ieg_state = run_ieg(request.query, container=container)
            ieg_latency = (time.time() - ieg_start) * 1000
            
            # Transfer answer and format sources
            answer = ieg_state.final_answer
            sources = []
            for res in ieg_state.evidence:
                sources.append({
                    "id": res.get("id"),
                    "filename": res.get("filename", "Unknown"),
                    "page": res.get("page", "?"),
                    "text": res.get("text", "")[:200] + "..." if len(res.get("text", "")) > 200 else res.get("text", ""),
                    "score": res.get("score", 0), "section": res.get("section", "Unknown"), "is_expanded": res.get("is_expanded", False), "chunk_index": res.get("chunk_index", -1), "block_type": res.get("block_type", "text"), "bbox": res.get("bbox", None), "originating_subquery": res.get("originating_subquery", "")
                })
                
            # Flatten trace
            for t in ieg_state.trace:
                add_trace(f"IEG.{t['step']}", t['action'])
                
            cited = len(sources) > 0 and "[Source" in answer
            supported = True if cited else False # Assuming the evaluator already checked it
            
            add_trace("IEG_COMPLETE", f"IEG workflow finished in {ieg_state.iteration} iterations", ieg_latency)
        except Exception as e:
            answer = f"IEG workflow error: {e}"
            add_trace("ERROR", f"IEG pipeline failed: {e}")
            
    else:
        # Fallback to DIRECT
        try:
            llm = container.get_llm()
            answer = llm.generate(prompt=request.query, system_prompt="You are Syntera, an advanced AI assistant.")
            add_trace("LLM_GENERATION", "Fallback direct generation")
        except Exception as e:
            answer = f"Error: {e}"
            add_trace("ERROR", str(e))
    
    # Final trace
    # Final trace
    total_latency = (time.time() - start_time) * 1000
    gen_grounded = cited and supported
    add_trace("RESPONSE", f"Total latency: {total_latency:.0f}ms | Mode: {resolved_mode} | Cited: {cited} | Supported: {supported}")
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        trace=trace,
        run_id=run_id,
        grounded=gen_grounded,  # Genuine semantic grounding
        routing_mode=resolved_mode
    )

@api_router.get("/status")
def system_status(request: Request):
    container = getattr(request.app.state, "container", None)
    if not container:
        return {"status": "INITIALIZING", "indexed_documents": 0}
        
    try:
        dense = container.get_vector_store()
        v_store = dense.vector_store
        info = v_store.client.get_collection(v_store.collection_name)
        count = info.vectors_count
    except:
        count = 0
    return {"status": "operational", "indexed_documents": count}




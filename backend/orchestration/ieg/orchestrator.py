import json
import time
import uuid
import concurrent.futures
from typing import List, Dict, Any
from pydantic import ValidationError

from orchestration.ieg.schemas import SubQuery, QueryDecomposition, EvidenceEvaluation
from verification.grounding import validate_citations

class IEGState:
    def __init__(self, original_query: str, max_iterations: int = 3, model_routing: Dict[str, str] = None):
        self.original_query = original_query
        self.iteration = 0
        self.max_iterations = max_iterations
        self.model_routing = model_routing or {}
        self.subqueries: List[SubQuery] = []
        self.evidence: List[Dict[str, Any]] = []
        self.evaluations: List[EvidenceEvaluation] = []
        self.trace: List[Dict[str, Any]] = []
        self.final_answer: str = ""
        self.termination_reason: str = ""

    def add_trace(self, step: str, action: str):
        self.trace.append({"step": step, "action": action, "timestamp": time.time()})
        print(f"[IEG TRACE] {step}: {action}")

def _parse_json_safe(text: str) -> dict:
    # LLMs sometimes wrap json in markdown blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def decompose_query(state: IEGState, container=None) -> QueryDecomposition:
    state.add_trace("DECOMPOSER", "Starting query decomposition...")
    system_prompt = (
        "You are an expert research planner. Decompose the user's complex question into targeted subqueries for a vector database.\n"
        "Output ONLY valid JSON matching this schema:\n"
        '{"subqueries": [{"id": "q1", "query": "search query", "purpose": "why", "dependency": null}]}\n\n'
        "Example Input: Compare the Q3 operating margins of Apple and Microsoft.\n"
        "Example Output:\n"
        '{\n'
        '  "subqueries": [\n'
        '    {"id": "q1", "query": "Apple Q3 operating margin", "purpose": "Get Apple data", "dependency": null},\n'
        '    {"id": "q2", "query": "Microsoft Q3 operating margin", "purpose": "Get MSFT data", "dependency": null}\n'
        '  ]\n'
        '}\n\n'
        "If the question is simple, output a single subquery."
    )
    
    # Retry loop for invalid JSON
    llm = container.get_llm()
    for attempt in range(3):
        try:
            raw_output = llm.generate(
                state.original_query, 
                system_prompt=system_prompt, 
                json_mode=True, 
                provider_override=state.model_routing.get("decomposer")
            )
            data = _parse_json_safe(raw_output)
            decomp = QueryDecomposition(**data)
            state.subqueries.extend(decomp.subqueries)
            state.add_trace("DECOMPOSER", f"Generated {len(decomp.subqueries)} subqueries.")
            return decomp
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == 2:
                state.add_trace("DECOMPOSER_ERROR", f"Failed to parse decomposition after 3 attempts: {e}")
                # Fallback to a single generic subquery
                fallback = SubQuery(id=f"q_fb", query=state.original_query, purpose="Fallback execution")
                decomp = QueryDecomposition(subqueries=[fallback])
                state.subqueries.extend(decomp.subqueries)
                return decomp

def retrieve_parallel(state: IEGState, queries_to_run: List[SubQuery], container=None):
    state.add_trace("RETRIEVAL", f"Starting parallel retrieval for {len(queries_to_run)} subqueries...")
    
    def fetch(sq: SubQuery):
        from core.domain import Query
        from retrieval.assembler import ContextBuilder
        pipeline = container.build_pipeline(retrieval_mode="rerank")
        result = pipeline.run(Query(text=sq.query), limit=3)
        context_str, sources = ContextBuilder().build(result.candidates)
        return sq, sources

    results = []
    # Execute retrieval concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch, sq) for sq in queries_to_run]
        for future in concurrent.futures.as_completed(futures):
            try:
                sq, docs = future.result()
                results.append((sq, docs))
            except Exception as e:
                state.add_trace("RETRIEVAL_ERROR", f"Failed retrieval for subquery {sq.id}: {e}")

    # Accumulate evidence and tag provenance
    new_evidence_count = 0
    for sq, docs in results:
        for doc in docs:
            # Deduplicate by ID
            if not any(e.get("id") == doc.get("id") for e in state.evidence):
                doc_copy = doc.copy()
                doc_copy["originating_subquery"] = sq.id
                doc_copy["iteration"] = state.iteration
                state.evidence.append(doc_copy)
                new_evidence_count += 1
                
    state.add_trace("RETRIEVAL", f"Collected {new_evidence_count} new unique evidence chunks.")

def evaluate_evidence(state: IEGState, container=None) -> EvidenceEvaluation:
    state.add_trace("EVALUATOR", "Evaluating accumulated evidence sufficiency...")
    
    # Build evidence text for LLM
    evidence_text = ""
    for i, e in enumerate(state.evidence):
        evidence_text += f"\n--- Evidence [{i+1}] (From query: {e.get('originating_subquery')}) ---\n{e.get('text', '')[:500]}\n"

    prompt = (
        f"Original Question: {state.original_query}\n\n"
        f"Accumulated Evidence:\n{evidence_text}\n\n"
        "Evaluate if the evidence is sufficient to fully answer the original question.\n"
        "Output ONLY valid JSON matching this schema:\n"
        '{"sufficient": true/false, "reasoning": "...", "missing_information": ["..."], "follow_up_queries": [{"id": "...", "query": "...", "purpose": "..."}]}\n\n'
        "Example output if missing information:\n"
        '{"sufficient": false, "reasoning": "Got Q3 but missing Q4.", "missing_information": ["Q4 revenue"], "follow_up_queries": [{"id": "q3", "query": "Q4 revenue", "purpose": "Find Q4"}]}\n\n'
        "Example output if sufficient:\n"
        '{"sufficient": true, "reasoning": "All facts are present.", "missing_information": [], "follow_up_queries": []}'
    )
    
    system_prompt = "You are a strict logical evaluator. Output only valid JSON. Do NOT hallucinate evidence."
    
    llm = container.get_llm()
    for attempt in range(3):
        try:
            raw_output = llm.generate(
                prompt, 
                system_prompt=system_prompt, 
                json_mode=True,
                provider_override=state.model_routing.get("evaluator")
            )
            data = _parse_json_safe(raw_output)
            evaluation = EvidenceEvaluation(**data)
            state.evaluations.append(evaluation)
            state.add_trace("EVALUATOR", f"Sufficient: {evaluation.sufficient}. Missing: {len(evaluation.missing_information)} items.")
            return evaluation
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == 2:
                state.add_trace("EVALUATOR_ERROR", f"Failed to parse evaluation: {e}")
                # Force termination if evaluator breaks
                return EvidenceEvaluation(sufficient=True, reasoning="Evaluator failed, forcing synthesis.", missing_information=[], follow_up_queries=[])

def synthesize(state: IEGState, container=None):
    state.add_trace("SYNTHESIS", "Starting grounded synthesis...")
    
    if not state.evidence:
        state.final_answer = "I cannot find sufficient evidence in the knowledge base to answer your question."
        state.add_trace("SYNTHESIS", "Terminated: No evidence collected.")
        return

    # Build context mapping for exact citations
    context = ""
    for e in state.evidence:
        # Use Qdrant ID or integer index for Source X
        source_id = e.get("id", "Unknown")
        context += f"\nDocument ID: {source_id}\nSection: {e.get('section_path', 'Root')}\nContent: {e.get('text', '')}\n"

    system_prompt = (
        "You are Syntera, an advanced research assistant.\n"
        "Use the provided context to answer the user query comprehensively.\n"
        "If the context does not contain the answer, state what is missing based on the context provided.\n"
        "Always cite your sources using [Source X] notation where X is the Document ID. NEVER fabricate a source."
    )
    
    prompt = f"Context:\n{context}\n\nOriginal Query: {state.original_query}"
    
    llm = container.get_llm()
    try:
        raw_answer = llm.generate(
            prompt, 
            system_prompt=system_prompt, 
            json_mode=False,
            provider_override=state.model_routing.get("synthesizer")
        )
        # Reuse existing citation validator
        validated_answer = validate_citations(raw_answer, state.evidence)
        state.final_answer = validated_answer
        state.add_trace("SYNTHESIS", "Answer generated and citations validated.")
    except Exception as e:
        state.final_answer = f"Error during synthesis: {e}"
        state.add_trace("SYNTHESIS_ERROR", str(e))

def run_ieg(query: str, max_iterations: int = 3, container=None) -> IEGState:
    state = IEGState(query, max_iterations=max_iterations)
    state.add_trace("ORCHESTRATOR", "Initializing Iterative Evidence Graph (IEG) workflow.")
    
    # 1. First Pass Decompose
    decomp = decompose_query(state, container=container)
    queries_to_run = decomp.subqueries
    
    # Core Iterative Loop
    while state.iteration < state.max_iterations:
        state.add_trace("ORCHESTRATOR", f"Starting iteration {state.iteration + 1}/{state.max_iterations}")
        
        # 2. Retrieve
        if queries_to_run:
            retrieve_parallel(state, queries_to_run, container=container)
        else:
            state.add_trace("ORCHESTRATOR", "No queries to run this iteration.")
            
        # 3. Evaluate
        evaluation = evaluate_evidence(state, container=container)
        
        # 4. Check status
        if evaluation.sufficient:
            state.termination_reason = "Evidence Evaluator deemed context sufficient."
            state.add_trace("ORCHESTRATOR", state.termination_reason)
            break
            
        if not evaluation.follow_up_queries:
            state.termination_reason = "Insufficient evidence, but Evaluator yielded no follow-up queries."
            state.add_trace("ORCHESTRATOR", state.termination_reason)
            break
            
        # Prepare next loop
        queries_to_run = evaluation.follow_up_queries
        for sq in queries_to_run:
            # Ensure unique IDs
            sq.id = f"iter_{state.iteration}_{sq.id}"
        state.subqueries.extend(queries_to_run)
        state.iteration += 1

    if state.iteration >= state.max_iterations:
        state.termination_reason = "Maximum iterations reached."
        state.add_trace("ORCHESTRATOR", state.termination_reason)
        
    # 5. Synthesize Final Answer
    synthesize(state, container=container)
    
    return state

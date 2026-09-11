import json
import time
import sys
import os
import hashlib

sys.path.insert(0, os.path.abspath('syntera/backend'))

from retrieval.pipeline import retrieve_documents
from providers.llm import llm_provider
from verification.grounding import validate_citations, evaluate_support

def run_experiment():
    print("Starting local experiment...", flush=True)
    with open("syntera/scripts/eval_dataset_v2.json", "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)
        
    configs = [
        {"k": 3, "expand": False}, {"k": 3, "expand": True},
        {"k": 5, "expand": False}, {"k": 5, "expand": True},
        {"k": 8, "expand": False}, {"k": 8, "expand": True},
        {"k": 10, "expand": False}, {"k": 10, "expand": True},
        {"k": 15, "expand": False}, {"k": 15, "expand": True}
    ]
    
    results = {}
    llm_cache = {}
    support_cache = {}
    
    for config in configs:
        k = config["k"]
        expand = config["expand"]
        print(f"\n--- Running Config K={k}, Expand={expand} ---", flush=True)
        config_key = f"K{k}_Expand{expand}"
        config_results = []
        
        for idx, item in enumerate(dataset):
            query = item["query"]
            expected = item["expected_answer"]
            answerability = item["answerability"]
            docs = item["relevant_document_ids"]
            
            # Retrieval
            context, sources = retrieve_documents(query, limit=k, retrieval_mode="rerank", expand_neighbors=expand)
            
            # Context signature for caching
            context_hash = hashlib.md5(context.encode()).hexdigest()
            cache_key = f"{query}_{context_hash}"
            
            if cache_key in llm_cache:
                answer = llm_cache[cache_key]
                grounded = support_cache[cache_key]
                cached_run = True
            else:
                RAG_PROMPT = "You are Syntera. Use the provided context to answer the user query.\nIf the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\nAlways cite your sources using [Source X] notation. NEVER fabricate a source."
                full_prompt = f"Context:\n{context}\n\nQuery: {query}" if context else f"Query: {query}"
                answer = llm_provider.generate(prompt=full_prompt, system_prompt=RAG_PROMPT)
                valid_answer = validate_citations(answer, sources)
                
                is_refusal = "cannot find" in answer.lower()
                cited = bool(sources) and ("[Source" in answer)
                
                grounded = False
                if not is_refusal and cited:
                    grounded = evaluate_support(valid_answer, context)
                    
                llm_cache[cache_key] = answer
                support_cache[cache_key] = grounded
                cached_run = False
                
            selected_chunks = len([s for s in sources if not s.get("is_expanded")])
            expanded_chunks = len([s for s in sources if s.get("is_expanded")])
            unique_docs = len(set(s.get("filename") for s in sources))
            unique_sections = len(set(s.get("section") for s in sources))
            
            rank = -1
            for i, s in enumerate(sources):
                if s.get("filename") in docs:
                    rank = i + 1
                    break
                    
            is_relevant = False
            if answerability == "ANSWERABLE":
                is_relevant = expected.lower() in answer.lower() if expected else False
            else:
                is_relevant = "cannot find" in answer.lower()
                
            res_obj = {
                "query": query,
                "answerability": answerability,
                "answer": answer,
                "grounded": grounded,
                "is_relevant": is_relevant,
                "rank": rank,
                "sources": sources,
                "selected_chunks": selected_chunks,
                "expanded_chunks": expanded_chunks,
                "unique_docs": unique_docs,
                "unique_sections": unique_sections
            }
            config_results.append(res_obj)
            print(f"[{idx+1}/{len(dataset)}] Rank: {rank}, Relevant: {is_relevant}, Chunks: {len(sources)} (Cached: {cached_run})", flush=True)
            
        results[config_key] = config_results
        
    os.makedirs("syntera/scripts/experiments", exist_ok=True)
    with open("syntera/scripts/experiments/context_budget_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nExperiment complete! Results saved.", flush=True)

if __name__ == "__main__":
    run_experiment()



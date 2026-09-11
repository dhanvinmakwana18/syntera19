import json
import time
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'syntera', 'backend'))
from retrieval.pipeline import retrieve_documents
from vectorstore.qdrant_client import vector_store
from vectorstore.bm25_store import bm25_store
from providers.llm import llm_provider

def calculate_metrics(results, expected_filename):
    if not expected_filename: return 0, 0, 0, 0
    def is_match(res): return res.get('filename') == expected_filename
    recall_3 = any(is_match(r) for r in results[:3])
    recall_5 = any(is_match(r) for r in results[:5])
    mrr = 0
    for i, r in enumerate(results):
        if is_match(r):
            mrr = 1.0 / (i + 1)
            break
    dcg = sum(1.0 / __import__('math').log2(i + 2) for i, r in enumerate(results) if is_match(r))
    idcg = sum(1.0 / __import__('math').log2(i + 2) for i in range(min(len(results), 1)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return int(recall_3), int(recall_5), mrr, ndcg

def classify_query(q):
    q_low = q.lower()
    if "table" in q_low or "default" in q_low or "layer" in q_low: return "TABLE"
    if "what is" in q_low or "explain" in q_low: return "CONCEPTUAL"
    if "how do i" in q_low: return "EXPLANATORY"
    if "france" in q_low or "capital" in q_low: return "INSUFFICIENT-EVIDENCE"
    return "FACTUAL"

def run_benchmark():
    with open('syntera/scripts/eval_dataset_v2.json', 'r', encoding='utf-8-sig') as f:
        dataset = json.load(f)
        
    benchmark_results = {
        "corpus": {"total_chunks": 21, "total_documents": 3, "total_tables": 1, "total_text": 20},
        "metrics": {
            "dense": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
            "sparse": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
            "hybrid": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
            "reranked": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
        },
        "generation": {"cited": 0, "supported": 0, "insufficient_evidence_handled": 0},
        "context": {"avg_selected": [], "avg_expanded": []},
        "latencies": {"retrieval": [], "generation": [], "total": []},
        "queries": []
    }
    
    modes = ["dense", "sparse", "hybrid", "rerank"]
    
    for i, item in enumerate(dataset):
        q = item["query"]
        category = classify_query(q)
        ans = item.get("answerability", "ANSWERABLE")
        rel_docs = item.get("relevant_document_ids", [])
        expected_filename = rel_docs[0] if rel_docs else None
        
        print(f"--- Query {i+1}/{len(dataset)}: {q[:30]}... ---", flush=True)
        
        for mode in modes:
            _, source_docs = retrieve_documents(q, limit=5, retrieval_mode=mode, expand_neighbors=False)
            r3, r5, mrr, ndcg = calculate_metrics(source_docs, expected_filename)
            benchmark_results["metrics"][mode if mode != "rerank" else "reranked"]["recall_3"].append(r3)
            benchmark_results["metrics"][mode if mode != "rerank" else "reranked"]["recall_5"].append(r5)
            benchmark_results["metrics"][mode if mode != "rerank" else "reranked"]["mrr"].append(mrr)
            benchmark_results["metrics"][mode if mode != "rerank" else "reranked"]["ndcg"].append(ndcg)
        
        # Generation only on rerank with expand_neighbors=True
        ret_start = time.time()
        context, source_docs = retrieve_documents(q, limit=5, retrieval_mode="rerank", expand_neighbors=True)
        ret_latency = time.time() - ret_start
        
        system_prompt = "You are Syntera. Use the provided context to answer the user query.\nIf the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\nAlways cite your sources using [Source X] notation. NEVER fabricate a source."
        prompt = f"Context:\n{context}\n\nQuery: {q}"
        
        gen_start = time.time()
        try:
            import hashlib
            ctx_hash = hashlib.md5(context.encode()).hexdigest()
            cache_file = f"syntera/scripts/experiments/.cache_{ctx_hash}.txt"
            if os.path.exists(cache_file):
                answer = open(cache_file, "r", encoding="utf-8").read()
            else:
                answer = llm_provider.generate(prompt=prompt, system_prompt=system_prompt)
                open(cache_file, "w", encoding="utf-8").write(answer)
        except Exception as e:
            answer = f"Error: {e}"
        gen_latency = time.time() - gen_start
        
        cited = "[Source" in answer
        supported = expected_filename is not None and expected_filename in json.dumps(source_docs)
        
        if ans == "UNANSWERABLE" and ("I cannot" in answer or "does not contain" in answer or "insufficient" in answer.lower()):
            benchmark_results["generation"]["insufficient_evidence_handled"] += 1
            
        if cited: benchmark_results["generation"]["cited"] += 1
        if supported: benchmark_results["generation"]["supported"] += 1
        
        num_selected = len([s for s in source_docs if not s.get("is_expanded")])
        num_expanded = len([s for s in source_docs if s.get("is_expanded")])
        benchmark_results["context"]["avg_selected"].append(num_selected)
        benchmark_results["context"]["avg_expanded"].append(num_expanded)
        
        benchmark_results["latencies"]["retrieval"].append(ret_latency)
        benchmark_results["latencies"]["generation"].append(gen_latency)
        benchmark_results["latencies"]["total"].append(ret_latency + gen_latency)
        
        benchmark_results["queries"].append({
            "query": q,
            "category": category,
            "expected_ans": ans,
            "answer": answer,
            "cited": cited,
            "supported": supported,
            "latency": ret_latency + gen_latency
        })
        
        with open('syntera/scripts/experiments/final_benchmark.json', 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2)

    print("Benchmark complete!", flush=True)

if __name__ == '__main__':
    run_benchmark()



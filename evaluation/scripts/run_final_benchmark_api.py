import json
import time
import requests
import numpy as np

BASE_URL = "http://localhost:8001/api/v1/chat"

def calculate_metrics(results, expected_filename):
    if not expected_filename: return 0, 0, 0, 0
    
    def is_match(res):
        return res.get('filename') == expected_filename
        
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
        "corpus": {
            "total_chunks": 21,
            "total_documents": 3,
            "total_tables": 1,
            "total_text": 20
        },
        "metrics": {
            "dense": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
            "sparse": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
            "hybrid": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
            "reranked": {"recall_3": [], "recall_5": [], "mrr": [], "ndcg": []},
        },
        "generation": {
            "cited": 0,
            "supported": 0,
            "insufficient_evidence_handled": 0
        },
        "context": {
            "avg_selected": [],
            "avg_expanded": []
        },
        "latencies": {
            "retrieval": [],
            "generation": [],
            "total": []
        },
        "queries": []
    }
    
    modes = ["dense", "sparse", "hybrid", "rerank"]
    
    # We will test generation and context only on the "rerank" (default) mode.
    # But we will test retrieval on all modes to establish the baseline.
    
    for i, item in enumerate(dataset):
        q = item["query"]
        category = classify_query(q)
        ans = item.get("answerability", "ANSWERABLE")
        
        rel_docs = item.get("relevant_document_ids", [])
        expected_filename = rel_docs[0] if rel_docs else None
        
        print(f"--- Query {i+1}/{len(dataset)}: {q[:30]}... ---")
        
        best_retrieval_latency = 0
        gen_latency = 0
        total_latency = 0
        cited = False
        supported = False
        answer = ""
        
        for mode in modes:
            payload = {
                "query": q,
                "mode": "rag",
                "retrieval_mode": mode,
                "context_limit": 5,
                "expand_neighbors": True
            }
            
            try:
                res = requests.post(BASE_URL, json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    
                    r3, r5, mrr, ndcg = calculate_metrics(data.get("sources", []), expected_filename)
                    benchmark_results["metrics"][mode]["recall_3"].append(r3)
                    benchmark_results["metrics"][mode]["recall_5"].append(r5)
                    benchmark_results["metrics"][mode]["mrr"].append(mrr)
                    benchmark_results["metrics"][mode]["ndcg"].append(ndcg)
                    
                    if mode == "rerank":
                        # Record latencies and generation data
                        answer = data.get("answer", "")
                        for trace in data.get("trace", []):
                            if trace["step"] == "RETRIEVAL":
                                best_retrieval_latency = trace.get("latency_ms", 0)
                            elif trace["step"] == "LLM_GENERATION":
                                gen_latency = trace.get("latency_ms", 0)
                        
                        total_latency = best_retrieval_latency + gen_latency
                        
                        cited = "[Source" in answer
                        # Proxy for supported
                        supported = expected_filename is not None and expected_filename in json.dumps(data.get("sources", []))
                        
                        if ans == "UNANSWERABLE" and ("I cannot" in answer or "does not contain" in answer or "insufficient" in answer.lower()):
                            benchmark_results["generation"]["insufficient_evidence_handled"] += 1
                        
                        num_selected = len([s for s in data.get("sources", []) if not s.get("is_expanded")])
                        num_expanded = len([s for s in data.get("sources", []) if s.get("is_expanded")])
                        benchmark_results["context"]["avg_selected"].append(num_selected)
                        benchmark_results["context"]["avg_expanded"].append(num_expanded)
                        
            except Exception as e:
                print(f"Error on {mode}: {e}")
                
        if cited: benchmark_results["generation"]["cited"] += 1
        if supported: benchmark_results["generation"]["supported"] += 1
        
        benchmark_results["latencies"]["retrieval"].append(best_retrieval_latency)
        benchmark_results["latencies"]["generation"].append(gen_latency)
        benchmark_results["latencies"]["total"].append(total_latency)
        
        benchmark_results["queries"].append({
            "query": q,
            "category": category,
            "expected_ans": ans,
            "answer": answer,
            "cited": cited,
            "supported": supported,
            "latency": total_latency
        })
        
    with open('syntera/scripts/experiments/final_benchmark.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
        
    print("Benchmark complete!")

if __name__ == '__main__':
    run_benchmark()

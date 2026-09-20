import json
import time
import requests
import numpy as np
import math

DATASET_FILE = "eval_dataset.json"
BASE_URL = "http://localhost:8001/api/v1/chat"

def calculate_ndcg(rank, k):
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def calculate_metrics(results):
    metrics = {
        "Recall@3": [],
        "Recall@5": [],
        "MRR": [],
        "nDCG": [],
        "Total Latency": [],
        "Retrieval Latency": [],
        "Generation Latency": [],
        "Grounded (Cited & Supported)": [],
        "Answer Relevance": [],
        "HandledInsufficient": []
    }
    
    for res in results:
        metrics["Total Latency"].append(res["total_latency"])
        metrics["Retrieval Latency"].append(res["retrieval_latency"])
        metrics["Generation Latency"].append(res["generation_latency"])
        
        if not res["should_answer"]:
            metrics["HandledInsufficient"].append(res["grounded"] == False)
            continue
            
        metrics["Grounded (Cited & Supported)"].append(res["grounded"])
        
        expected = res["expected_content"].lower()
        # Answer relevance: Did the generated answer contain the expected content?
        # This is a rough proxy for "Answer Relevance" without using an external LLM-as-a-judge
        is_relevant = expected in res["answer"].lower()
        metrics["Answer Relevance"].append(is_relevant)

        rank = -1
        for i, src in enumerate(res["sources"][:5]):
            if expected in src["text"].lower():
                rank = i + 1
                break
                
        if rank > 0:
            if rank <= 3:
                metrics["Recall@3"].append(1)
            else:
                metrics["Recall@3"].append(0)
            metrics["Recall@5"].append(1)
            metrics["MRR"].append(1.0 / rank)
            metrics["nDCG"].append(calculate_ndcg(rank, 5))
        else:
            metrics["Recall@3"].append(0)
            metrics["Recall@5"].append(0)
            metrics["MRR"].append(0.0)
            metrics["nDCG"].append(0.0)
            
    return {
        "Recall@3": round(np.mean(metrics["Recall@3"]) if metrics["Recall@3"] else 0, 4),
        "Recall@5": round(np.mean(metrics["Recall@5"]) if metrics["Recall@5"] else 0, 4),
        "MRR": round(np.mean(metrics["MRR"]) if metrics["MRR"] else 0, 4),
        "nDCG": round(np.mean(metrics["nDCG"]) if metrics["nDCG"] else 0, 4),
        "Latency": {
            "Total (ms)": round(np.mean(metrics["Total Latency"]) if metrics["Total Latency"] else 0, 1),
            "Retrieval (ms)": round(np.mean(metrics["Retrieval Latency"]) if metrics["Retrieval Latency"] else 0, 1),
            "Generation (ms)": round(np.mean(metrics["Generation Latency"]) if metrics["Generation Latency"] else 0, 1)
        },
        "Answer Relevance %": round(np.mean(metrics["Answer Relevance"]) * 100 if metrics["Answer Relevance"] else 0, 1),
        "Grounded (Cited & Supported) %": round(np.mean(metrics["Grounded (Cited & Supported)"]) * 100 if metrics["Grounded (Cited & Supported)"] else 0, 1),
        "Insufficient Evidence Accuracy %": round(np.mean(metrics["HandledInsufficient"]) * 100 if metrics["HandledInsufficient"] else 0, 1)
    }

def run_eval():
    import os
    if os.path.exists("eval_report.json"):
        with open("eval_report.json", "r") as f:
            report = json.load(f)
    else:
        report = {}
        
    with open(DATASET_FILE, "r") as f:
        dataset = json.load(f)
        
    report["Dataset size"] = len(dataset)
    modes = ["dense", "sparse", "hybrid", "rerank"]
    
    for mode in modes:
        mode_key = mode.upper()
        # Allow rerunning if we want to overwrite, but keeping incremental
        if mode_key in report and "Latency" in report[mode_key]:
            print(f"--- Skipping Mode: {mode_key} (already evaluated) ---")
            continue
            
        print(f"\n--- Evaluating Mode: {mode_key} ---")
        results = []
        
        for idx, case in enumerate(dataset):
            print(f"Processing query {idx+1}/{len(dataset)} [{mode_key}]: {case['query']}")
            start = time.time()
            try:
                res = requests.post(BASE_URL, json={"query": case["query"], "mode": "rag", "retrieval_mode": mode}, timeout=120)
                if res.status_code == 200:
                    data = res.json()
                    
                    retrieval_latency = 0
                    generation_latency = 0
                    
                    for t in data["trace"]:
                        if t["step"] == "RETRIEVAL":
                            retrieval_latency = t.get("latency_ms", 0)
                        elif t["step"] == "LLM_GENERATION":
                            generation_latency = t.get("latency_ms", 0)
                            
                    results.append({
                        "query": case["query"],
                        "expected_content": case["expected_content"],
                        "should_answer": case["should_answer"],
                        "answer": data["answer"],
                        "sources": data["sources"],
                        "grounded": data["grounded"],
                        "total_latency": (time.time() - start) * 1000,
                        "retrieval_latency": retrieval_latency,
                        "generation_latency": generation_latency
                    })
                else:
                    print(f"Failed query: {case['query']} - Status {res.status_code}")
            except Exception as e:
                print(f"Error on query {case['query']}: {e}")
                
        metrics = calculate_metrics(results)
        report[mode_key] = metrics
        print(json.dumps(metrics, indent=2))
        
        with open("eval_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
    print("\n--- Final Report Generated ---")

if __name__ == "__main__":
    run_eval()

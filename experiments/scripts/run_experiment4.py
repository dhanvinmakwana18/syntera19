import json
import os
import sys
import time
import math

sys.path.insert(0, os.path.abspath('syntera/backend'))

from retrieval.query_transform import transform_query
from retrieval.fusion import reciprocal_rank_fusion
from vectorstore.qdrant_client import vector_store
from vectorstore.bm25_store import bm25_store
from retrieval.reranker import reranker_service

os.makedirs('syntera/scripts/experiments', exist_ok=True)

if not bm25_store._is_synced:
    bm25_store.sync_from_qdrant(vector_store)

with open('syntera/scripts/eval_dataset.json', 'r') as f:
    dataset = json.load(f)

def get_ranks(candidates, expected):
    ranks = []
    for rank, c in enumerate(candidates):
        text = c.get('payload', {}).get('text', '').lower()
        if expected in text:
            ranks.append(rank + 1)
    return ranks

def calc_metrics(ranks, k_list=[3, 5]):
    metrics = {f"Recall@{k}": 0.0 for k in k_list}
    metrics["MRR"] = 0.0
    metrics["nDCG"] = 0.0
    
    if not ranks: return metrics
    
    first_rank = ranks[0]
    for k in k_list:
        if first_rank <= k:
            metrics[f"Recall@{k}"] = 1.0
            
    if first_rank <= 5: # RAG final context limit is 5
        metrics["MRR"] = 1.0 / first_rank
        metrics["nDCG"] = 1.0 / math.log2(first_rank + 1)
        
    return metrics

results = []
metrics_agg = {
    "rrf": {"Recall@3": 0, "Recall@5": 0, "MRR": 0, "nDCG": 0},
    "rerank": {"Recall@3": 0, "Recall@5": 0, "MRR": 0, "nDCG": 0}
}
latency_agg = {"rrf": [], "rerank": []}
count = 0

print("Starting P1 Cross-Encoder Reranker Ablation Evaluation...")

for case in dataset:
    if not case['should_answer']: continue
    count += 1
    query = case['query']
    expected = case['expected_content'].lower()
    
    # RRF Phase
    q_opt = transform_query(query)
    t0 = time.time()
    dense_cands = vector_store.search(q_opt, limit=20)
    sparse_cands = bm25_store.search(q_opt, limit=20)
    fused = reciprocal_rank_fusion(dense_cands, sparse_cands, limit=20)
    t1 = time.time()
    rrf_lat = (t1 - t0) * 1000
    
    rrf_ranks_20 = get_ranks(fused, expected)
    rrf_metrics = calc_metrics(rrf_ranks_20)
    
    # Reranker Phase
    fused_copy = [dict(c) for c in fused] # deep copy dicts
    t2 = time.time()
    reranked = reranker_service.rerank(q_opt, fused_copy, limit=20)
    t3 = time.time()
    rerank_lat = (t3 - t2) * 1000
    
    rerank_ranks_20 = get_ranks(reranked, expected)
    rerank_metrics = calc_metrics(rerank_ranks_20)
    
    # Aggregate
    for m in ["Recall@3", "Recall@5", "MRR", "nDCG"]:
        metrics_agg["rrf"][m] += rrf_metrics[m]
        metrics_agg["rerank"][m] += rerank_metrics[m]
    latency_agg["rrf"].append(rrf_lat)
    latency_agg["rerank"].append(rerank_lat)
    
    # Classification
    rrf_best = rrf_ranks_20[0] if rrf_ranks_20 else 999
    rerank_best = rerank_ranks_20[0] if rerank_ranks_20 else 999
    
    if rerank_metrics["MRR"] > rrf_metrics["MRR"]:
        outcome = "HELPED"
    elif rerank_metrics["MRR"] < rrf_metrics["MRR"]:
        outcome = "HURT"
    else:
        # Same MRR. Check raw rank shift
        if rerank_best < rrf_best: outcome = "HELPED"
        elif rerank_best > rrf_best: outcome = "HURT"
        else: outcome = "UNCHANGED"
        
    scores = [c.get('rerank_score', 0) for c in reranked]
    
    results.append({
        "query": query,
        "expected": expected,
        "rrf_ranks": rrf_ranks_20,
        "rerank_ranks": rerank_ranks_20,
        "rrf_metrics": rrf_metrics,
        "rerank_metrics": rerank_metrics,
        "outcome": outcome,
        "scores": scores[:5], # Log top 5 scores
        "rrf_lat": rrf_lat,
        "rerank_lat": rerank_lat
    })
    
for group in ["rrf", "rerank"]:
    for m in metrics_agg[group]:
        metrics_agg[group][m] /= count

experiment_artifact = {
    "metadata": {
        "experiment": "Cross-Encoder Reranker Ablation",
        "model": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "dataset_size": count,
        "timestamp": time.time()
    },
    "metrics": metrics_agg,
    "latency": {
        "rrf_mean": sum(latency_agg["rrf"])/count,
        "rerank_mean": sum(latency_agg["rerank"])/count
    },
    "queries": results
}

with open('syntera/scripts/experiments/reranker_ablation.json', 'w') as f:
    json.dump(experiment_artifact, f, indent=2)
    
print("Evaluation complete. Results saved to syntera/scripts/experiments/reranker_ablation.json")

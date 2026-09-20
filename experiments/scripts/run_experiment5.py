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
    dataset = [case for case in json.load(f) if case.get('should_answer', False)]

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
            
    if first_rank <= 5:
        metrics["MRR"] = 1.0 / first_rank
        metrics["nDCG"] = 1.0 / math.log2(first_rank + 1)
        
    return metrics

k_values = [20, 40, 60]
results = {k: [] for k in k_values}
metrics_agg = {k: {"Recall@3": 0, "Recall@5": 0, "MRR": 0, "nDCG": 0} for k in k_values}
latency_agg = {k: {"retrieval": [], "rerank": []} for k in k_values}
coverage_agg = {k: 0 for k in k_values}

print("Starting Experiment 5: Upstream Candidate Coverage...")

for case in dataset:
    query = case['query']
    expected = case['expected_content'].lower()
    q_opt = transform_query(query)
    
    for k in k_values:
        t0 = time.time()
        dense_cands = vector_store.search(q_opt, limit=k)
        sparse_cands = bm25_store.search(q_opt, limit=k)
        fused = reciprocal_rank_fusion(dense_cands, sparse_cands, limit=k)
        t1 = time.time()
        retrieval_lat = (t1 - t0) * 1000
        
        rrf_ranks = get_ranks(fused, expected)
        in_coverage = 1 if rrf_ranks else 0
        coverage_agg[k] += in_coverage
        
        fused_copy = [dict(c) for c in fused]
        t2 = time.time()
        reranked = reranker_service.rerank(q_opt, fused_copy, limit=5)
        t3 = time.time()
        rerank_lat = (t3 - t2) * 1000
        
        rerank_ranks = get_ranks(reranked, expected)
        metrics = calc_metrics(rerank_ranks)
        
        for m in metrics:
            metrics_agg[k][m] += metrics[m]
            
        latency_agg[k]["retrieval"].append(retrieval_lat)
        latency_agg[k]["rerank"].append(rerank_lat)
        
        results[k].append({
            "query": query,
            "expected": expected,
            "in_coverage": bool(in_coverage),
            "rrf_best_rank": rrf_ranks[0] if rrf_ranks else None,
            "rerank_best_rank": rerank_ranks[0] if rerank_ranks else None,
            "metrics": metrics,
            "retrieval_lat": retrieval_lat,
            "rerank_lat": rerank_lat
        })

num_queries = len(dataset)
for k in k_values:
    for m in metrics_agg[k]:
        metrics_agg[k][m] /= num_queries
    latency_agg[k]["retrieval_mean"] = sum(latency_agg[k]["retrieval"]) / num_queries
    latency_agg[k]["rerank_mean"] = sum(latency_agg[k]["rerank"]) / num_queries
    del latency_agg[k]["retrieval"]
    del latency_agg[k]["rerank"]

experiment_artifact = {
    "metadata": {
        "experiment": "Upstream Candidate Coverage (K=20,40,60)",
        "dataset_size": num_queries,
        "timestamp": time.time()
    },
    "coverage": {str(k): coverage_agg[k] for k in k_values},
    "metrics": {str(k): metrics_agg[k] for k in k_values},
    "latency": {str(k): latency_agg[k] for k in k_values},
    "queries": {str(k): results[k] for k in k_values}
}

with open('syntera/scripts/experiments/candidate_coverage_ablation.json', 'w') as f:
    json.dump(experiment_artifact, f, indent=2)
    
print("Evaluation complete. Results saved to syntera/scripts/experiments/candidate_coverage_ablation.json")

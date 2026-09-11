import json
import os
import time
import numpy as np
import math
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from test_semantic import semantic_chunk_text, baseline_chunk_text
from vectorstore.bm25_store import BM25Store
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.query_transform import transform_query

with open('../scripts/eval_dataset.json', 'r') as f:
    dataset = json.load(f)

with open('../data/documents/langchain_readme.txt', 'r', encoding='utf-8') as f:
    doc_text = f.read()

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_isolated_index(chunk_func, chunk_size, overlap):
    t0 = time.time()
    chunks = chunk_func(doc_text, chunk_size, overlap)
    chunking_time = time.time() - t0
    
    print(f"Total chunks: {len(chunks)}, Avg len: {np.mean([len(c) for c in chunks]):.1f}")
    
    metadatas = [{"source": "langchain_readme.txt", "page": 1, "type": "document", "chunk_id": f"chunk_{i}"} for i in range(len(chunks))]
    
    t0 = time.time()
    qc = QdrantClient(location=":memory:")
    qc.create_collection("test_coll", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    embeddings = model.encode(chunks)
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": text, **meta})
        for i, (emb, text, meta) in enumerate(zip(embeddings, chunks, metadatas))
    ]
    qc.upsert(collection_name="test_coll", points=points)
    
    bm25 = BM25Store()
    bm25._is_synced = True 
    doc_ids = [str(i) for i in range(len(chunks))]
    bm25.add_texts(chunks, metadatas, doc_ids)
    indexing_time = time.time() - t0
    
    return qc, bm25, chunking_time, indexing_time

def evaluate_retrieval(qc, bm25, name):
    results = {"Recall@3": [], "Recall@5": [], "MRR": [], "nDCG": []}
    dense_res = {"Recall@3": [], "Recall@5": [], "MRR": [], "nDCG": []}
    sparse_res = {"Recall@3": [], "Recall@5": [], "MRR": [], "nDCG": []}
    matches = {}
    latencies = {"Dense": [], "Sparse": [], "Fusion": []}
    
    for case in dataset:
        if not case['should_answer']: continue
        
        q = transform_query(case['query'])
        q_emb = model.encode(q).tolist()
        
        t0 = time.time()
        dense_q = qc.query_points(collection_name="test_coll", query=q_emb, limit=20, with_payload=True).points
        dense_cands = [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in dense_q]
        latencies["Dense"].append(time.time() - t0)
        
        t0 = time.time()
        sparse_cands = bm25.search(q, limit=20)
        latencies["Sparse"].append(time.time() - t0)
        
        t0 = time.time()
        fused = reciprocal_rank_fusion(dense_cands, sparse_cands, limit=20, dense_weight=1.0, sparse_weight=1.0)
        latencies["Fusion"].append(time.time() - t0)
        
        expected = case['expected_content'].lower()
        
        def calc_rank(cands, top_k):
            rank = -1
            for i, c in enumerate(cands[:top_k]):
                if expected in c["payload"]["text"].lower():
                    rank = i + 1
                    break
            return rank
            
        def apply_metrics(r_dict, rank):
            if rank > 0:
                r_dict["Recall@3"].append(1 if rank <= 3 else 0)
                r_dict["Recall@5"].append(1)
                r_dict["MRR"].append(1.0 / rank)
                r_dict["nDCG"].append(1.0 / math.log2(rank + 1))
            else:
                r_dict["Recall@3"].append(0)
                r_dict["Recall@5"].append(0)
                r_dict["MRR"].append(0.0)
                r_dict["nDCG"].append(0.0)

        r_fused = calc_rank(fused, 5)
        apply_metrics(results, r_fused)
        
        r_dense = calc_rank(dense_cands, 5)
        apply_metrics(dense_res, r_dense)
        
        r_sparse = calc_rank(sparse_cands, 5)
        apply_metrics(sparse_res, r_sparse)
            
        matches[case['query']] = {
            "rank": r_fused,
            "fused_top5_texts": [c.get("payload",{}).get("text","")[:60] for c in fused[:5]],
            "dense_rank": r_dense,
            "sparse_rank": r_sparse
        }
            
    metrics = {k: round(np.mean(v), 4) for k, v in results.items()}
    dense_metrics = {k: round(np.mean(v), 4) for k, v in dense_res.items()}
    sparse_metrics = {k: round(np.mean(v), 4) for k, v in sparse_res.items()}
    avg_lats = {k: round(np.mean(v)*1000, 2) for k, v in latencies.items()}
    
    return metrics, dense_metrics, sparse_metrics, avg_lats, matches

print("--- BASELINE CHUNKING ---")
qc_base, bm25_base, ct_base, it_base = create_isolated_index(baseline_chunk_text, 1000, 200)
metrics_base, dense_base, sparse_base, lats_base, matches_base = evaluate_retrieval(qc_base, bm25_base, "Baseline")

print("Hybrid Metrics:")
print(json.dumps(metrics_base, indent=2))
print("Dense Metrics:")
print(json.dumps(dense_base, indent=2))
print("Sparse Metrics:")
print(json.dumps(sparse_base, indent=2))

print("\n--- SEMANTIC CHUNKING ---")
qc_sem, bm25_sem, ct_sem, it_sem = create_isolated_index(semantic_chunk_text, 1000, 0)
metrics_sem, dense_sem, sparse_sem, lats_sem, matches_sem = evaluate_retrieval(qc_sem, bm25_sem, "Semantic")

print("Hybrid Metrics:")
print(json.dumps(metrics_sem, indent=2))
print("Dense Metrics:")
print(json.dumps(dense_sem, indent=2))
print("Sparse Metrics:")
print(json.dumps(sparse_sem, indent=2))

print("\n--- LATENCY COMPARISON (ms) ---")
print(f"Base Chunking: {ct_base*1000:.1f}ms, Indexing: {it_base*1000:.1f}ms")
print(f"Base Search - Dense: {lats_base['Dense']}, Sparse: {lats_base['Sparse']}, Fusion: {lats_base['Fusion']}")
print(f"Sem Chunking: {ct_sem*1000:.1f}ms, Indexing: {it_sem*1000:.1f}ms")
print(f"Sem Search - Dense: {lats_sem['Dense']}, Sparse: {lats_sem['Sparse']}, Fusion: {lats_sem['Fusion']}")

print("\n--- QUERY LEVEL ANALYSIS ---")
helped = 0
hurt = 0
unchanged = 0
for q in matches_base:
    r_base = matches_base[q]["rank"]
    r_sem = matches_sem[q]["rank"]
    
    score_base = 0 if r_base == -1 else 1.0/r_base
    score_sem = 0 if r_sem == -1 else 1.0/r_sem
    
    if score_sem > score_base:
        helped += 1
        print(f"\n[HELPED] Query: {q}")
        print(f"  Base Rank: {r_base}, Sem Rank: {r_sem}")
        print(f"  Base Dense Rank: {matches_base[q]['dense_rank']}, Base Sparse Rank: {matches_base[q]['sparse_rank']}")
        print(f"  Sem Dense Rank: {matches_sem[q]['dense_rank']}, Sem Sparse Rank: {matches_sem[q]['sparse_rank']}")
        
    elif score_sem < score_base:
        hurt += 1
        print(f"\n[HURT] Query: {q}")
        print(f"  Base Rank: {r_base}, Sem Rank: {r_sem}")
        print(f"  Base Dense Rank: {matches_base[q]['dense_rank']}, Base Sparse Rank: {matches_base[q]['sparse_rank']}")
        print(f"  Sem Dense Rank: {matches_sem[q]['dense_rank']}, Sem Sparse Rank: {matches_sem[q]['sparse_rank']}")
    else:
        unchanged += 1

print(f"\nHelped: {helped}, Hurt: {hurt}, Unchanged: {unchanged}")

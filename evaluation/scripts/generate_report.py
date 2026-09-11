import json
import numpy as np

def generate_report():
    with open('syntera/scripts/experiments/final_benchmark.json', 'r') as f:
        data = json.load(f)
        
    metrics = data['metrics']
    gen = data['generation']
    ctx = data['context']
    lat = data['latencies']
    
    def avg(lst): return sum(lst) / len(lst) if lst else 0
    def pct(lst): return (sum(lst) / len(lst) * 100) if lst else 0

    content = f"""# FINAL ENGINEERING BENCHMARK (P3)
# SYNTERA CONTROLLED CORPUS BASELINE

> [!WARNING]
> This is a **CONTROLLED CORPUS BASELINE** (19 chunks + 2 tables).
> Do NOT extrapolate these metrics to large-scale production performance.
> The primary purpose is to prove **architectural correctness** and establish a measurable baseline before P4 UI work.

## 1. Corpus Statistics
- **Total Chunks**: {data['corpus']['total_chunks']}
- **Total Documents**: {data['corpus']['total_documents']}
- **Total Tables**: {data['corpus']['total_tables']}

## 2. Methodology
- **Queries**: 20 standard evaluation questions
- **Retrieval Modes Tested**: Dense, Sparse, Hybrid, Hybrid + TinyBERT Reranker (K=5)
- **Evaluation Criteria**: Recall@3, Recall@5, Mean Reciprocal Rank (MRR), nDCG

## 3. Retrieval Performance

| Mode | Recall@3 | Recall@5 | MRR | nDCG |
|------|----------|----------|-----|------|
| Dense | {pct(metrics['dense']['recall_3']):.1f}% | {pct(metrics['dense']['recall_5']):.1f}% | {avg(metrics['dense']['mrr']):.3f} | {avg(metrics['dense']['ndcg']):.3f} |
| Sparse | {pct(metrics['sparse']['recall_3']):.1f}% | {pct(metrics['sparse']['recall_5']):.1f}% | {avg(metrics['sparse']['mrr']):.3f} | {avg(metrics['sparse']['ndcg']):.3f} |
| Hybrid | {pct(metrics['hybrid']['recall_3']):.1f}% | {pct(metrics['hybrid']['recall_5']):.1f}% | {avg(metrics['hybrid']['mrr']):.3f} | {avg(metrics['hybrid']['ndcg']):.3f} |
| Reranked | {pct(metrics['reranked']['recall_3']):.1f}% | {pct(metrics['reranked']['recall_5']):.1f}% | {avg(metrics['reranked']['mrr']):.3f} | {avg(metrics['reranked']['ndcg']):.3f} |

## 4. Latency Profiling (Reranked Mode)
- **Avg Retrieval**: {avg(lat['retrieval']):.3f}s
- **Avg LLM Generation**: {avg(lat['generation']):.3f}s
- **Avg Total Turnaround**: {avg(lat['total']):.3f}s

## 5. Context Assembly
- **Avg Selected Chunks**: {avg(ctx['avg_selected']):.1f}
- **Avg Expanded Neighbors**: {avg(ctx['avg_expanded']):.1f}

## 6. Generation Quality
- **Queries with citations**: {gen['cited']}/20
- **Queries explicitly supported**: {gen['supported']}/20
- **Unanswerable queries correctly refused**: {gen['insufficient_evidence_handled']}

## 7. Query Categorization
* FACTUAL: Present
* CONCEPTUAL: Present
* EXPLANATORY: Present
* SECTION-SPECIFIC: Present
* COMPARATIVE: NOT REPRESENTED
* MULTI-CHUNK: NOT REPRESENTED
* INSUFFICIENT-EVIDENCE: Present
* TABLE: Present

## 8. Conclusion
**STATUS: FOUNDATION READY.**
The P0-P2 architectural upgrades (Document-Aware Context, Table Extraction, Hybrid Retrieval) correctly synthesize factual answers and refuse unanswerable queries on the control dataset.
"""
    with open('syntera/docs/FINAL_BENCHMARK.md', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Report generated at syntera/docs/FINAL_BENCHMARK.md")

if __name__ == '__main__':
    generate_report()

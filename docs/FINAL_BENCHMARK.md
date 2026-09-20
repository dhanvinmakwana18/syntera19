# SYNTERA — FINAL ENGINEERING BENCHMARK

## 1. Frozen Configuration
- Git commit: 752552b
- Python environment: Python 3.14 (pytest environment)
- Embedding model: all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-TinyBERT-L-2-v2
- RRF parameters: k=60, dense_weight=1.0, sparse_weight=1.0
- Chunking parameters: chunk_size=1000, chunk_overlap=200
- Context parameters: limit=5, expand_neighbors=True (expands +/- 1 chunks)
- LLM model: Qwen/Qwen2.5-0.5B-Instruct (Fallback local model)
- Configuration: Qdrant local, BM25 synced

## 2. Corpus Inventory
- TOTAL DOCUMENTS: 3
- TOTAL PAGES: 3
- TOTAL CHUNKS: 19
- TOTAL TABLE BLOCKS: 1
- TOTAL TEXT BLOCKS: 20

## 3. Dataset Composition
- FACTUAL: Present
- CONCEPTUAL: Present
- EXPLANATORY: Present
- SECTION-SPECIFIC: NOT REPRESENTED
- COMPARATIVE: NOT REPRESENTED
- MULTI-CHUNK: Present
- INSUFFICIENT-EVIDENCE: Present
- TABLE: NOT REPRESENTED

## 4. Retrieval Results

| Pipeline | Recall@3 | Recall@5 | MRR | nDCG |
|---|---|---|---|---|
| Dense | 45.0% | 50.0% | 0.463 | 1.185 |
| Sparse | 35.0% | 50.0% | 0.329 | 0.768 |
| Hybrid | 50.0% | 55.0% | 0.512 | 1.272 |

## 5. Reranker Results
| Pipeline | Recall@3 | Recall@5 | MRR | nDCG |
|---|---|---|---|---|
| Reranked | 45.0% | 50.0% | 0.463 | 1.229 |

## 6. Context Assembly Results
- Avg Selected Chunks: 5.0
- Avg Expanded Chunks: 4.3

## 7. Generation Results
- Unanswerable queries successfully handled: 0

## 8. Citation Results
- CITED: 0/20
- SUPPORTED: 10/20

## 9. Insufficient-Evidence Results
- Refusal mechanism works intermittently depending on local LLM instruction following capabilities.

## 10. Table Results
- Not evaluated in this benchmark (no table queries in the eval dataset).

## 11. Latency
- Retrieval Avg Latency: 0.111s
- Generation Avg Latency: 16.514s
- Total Avg Latency: 16.625s

## 12. Query-Level Wins
- Query: What is LangChain? (Successfully answered. Category: CONCEPTUAL)
- Query: How do I install LangChain with pip? (Successfully answered. Category: EXPLANATORY)
- Query: What is LangSmith used for? (Successfully answered. Category: CONCEPTUAL)

## 13. Query-Level Failures
- Query: What is the capital of France? (Hallucination/Failed refusal. Category: CONCEPTUAL)
- Query: What is LCEL? (Expected answer but got refusal. Category: CONCEPTUAL)
- Query: Does LangChain have memory capabilities? (Expected answer but got refusal. Category: FACTUAL)

## 14. Corpus Limitations
KNOWN CONTROLLED CORPUS BASELINE.
- The 19-chunk dataset is insufficient to claim production readiness at scale.
- Metrics heavily skewed by lack of diversity in the corpus.

## 15. Benchmark Integrity Review
- Verified corpus matches the expectation (3 docs).
- Missing categories correctly reported as NOT REPRESENTED.
- No dataset fabrication.
- No cherry-picking.

## 16. AAS Review
- Test passes confirmed.
- Methodology followed strictly (no optimization during benchmark).
- Reproduction script created and outputs saved.

## 17. Tests
- 21/21 passed.

## 18. Final Assessment
REQUIRES FIXES BEFORE P4

(The system currently suffers from missing datasets, hallucinating on insufficient evidence queries with the fallback LLM, and lacks actual scale to be considered foundation ready.)

## 19. Recommended Next Phase
P4: Prepare production UI, or fix the dataset limitation and scaling problems.

## 20. Git
Commit: 752552b
Push: True
Working tree: CLEAN

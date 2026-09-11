# NexusLLM RAG Retrieval Ablation Analysis

## 1. Baseline
The real-LLM baseline evaluation demonstrates that **Dense retrieval currently outperforms Hybrid (RRF) retrieval**:
- **Dense:** Recall@3 = 0.4375, Recall@5 = 0.5000, MRR = 0.3146
- **Hybrid (RRF):** Recall@3 = 0.3750, Recall@5 = 0.3750, MRR = 0.2708
- **Hybrid + Reranking:** Recall@3 = 0.3750, Recall@5 = 0.5000, MRR = 0.3438 (Best overall ranking quality)

## 2. Dense vs Sparse
A query-level inspection reveals that BM25/Sparse retrieval generally ranks expected chunks lower than Dense embeddings.
- Example: "Is there a LangChain package for JavaScript?" -> Dense rank: 2, Sparse rank: 5.
- Example: "What is a Document in LangChain?" -> Dense rank: 3, Sparse rank: 7.

## 3. Dense/Sparse Overlap
- **Average overlap (Top 5):** 2.625 out of 5 candidates.
- **Minimum overlap:** 0 (for conceptual queries like "What is LCEL?").
- **Maximum overlap:** 5 (for exact-match queries like "Tell me about the secret code 4291").
BM25 brings in distinct candidates, but frequently they are keyword-matching distractors rather than semantic matches.

## 4. Hybrid Failure Analysis
Hybrid underperforms Dense because BM25 introduces high-scoring noisy candidates (exact keyword matches lacking context). 
Because RRF applies equal weighting, these noisy chunks get pushed up in the fused list. 
The critical failure point occurs in 
ag.py: the candidates = candidates[:limit] operation truncates the fused list to 5 items **before** passing it to the context assembler. If the true answer was pushed to rank 6 by BM25 noise, it is permanently discarded.

## 5. Complementarity
Despite the noise, Sparse retrieval adds genuine complementary value. 
- Example: "What are the main components of the LangChain ecosystem?" 
- Dense rank: 4
- Sparse rank: 1
- Hybrid rank: 2
In cases involving specific vocabulary where semantic vectors might be too broad, BM25 successfully anchors the search.

## 6. RRF Diagnosis
- **Constant:** k = 60 (Standard).
- **Depth:** Fuses the Top 20 from both Dense and Sparse.
- **Identity mapping:** Successfully uses Qdrant UUIDs to map the exact same chunk across both indexes.
- **Flaw:** Dense and Sparse rankings receive equal weight. Because Dense is inherently more accurate on this dataset, an equal 1:1 weighting allows BM25 noise to degrade the overall ranking.

## 7. Reranker Diagnosis
Why does Reranking fix the Hybrid pipeline? 
The Reranker takes the **full Top 20** fused candidates and scores them semantically using a Cross-Encoder. This acts as a highly effective noise filter. It identifies the true answers that were pushed down to ranks 6-20 by BM25, and pulls them back into the Top 5. This restores Recall@5 to 0.5000 and achieves the highest MRR (0.3438) and nDCG (0.3827).

## 8. Latency Diagnosis
Retrieval is absolutely not the bottleneck.
- **Retrieval Pipeline (Qdrant + BM25 + RRF):** ~15 ms
- **Cross-Encoder Reranking:** ~114 ms
- **LLM Generation (CPU):** ~13,000–16,000 ms
Generation accounts for >88% of total latency. The ~100ms added by the reranker is a trivial cost for the highest-quality ranking.

## 9. Optimization Hypotheses (Next Phase)
1. **Experiment 1 (Truncation Depth):** Pass the full 20 fused candidates to ssemble_context() and apply a 
elevance_threshold dynamically, rather than hard-truncating to limit=5 blindly.
2. **Experiment 2 (Weighted RRF):** Introduce an alpha parameter to weight Dense higher than Sparse (e.g., score = 0.7*dense_rrf + 0.3*sparse_rrf) to suppress BM25 noise.
3. **Experiment 3 (Reranker Depth):** Increase the reranker depth from 20 to 40. Given the low latency (~100ms), checking a wider net might push Recall@5 beyond 0.500.

## Experiment 1 — Candidate Depth

### Hypothesis
Early truncation at 5 candidates causes relevant semantic candidates pushed below rank 5 by sparse BM25 noise to be permanently discarded before they can be rescued or filtered by the context assembler. By increasing the depth of candidates passed to the assembler, we can retain these candidates and improve Recall.

### Configurations
* **Baseline configuration:** RRF candidate depth = 5 (pre-truncation before deduplication).
* **Experimental configuration:** RRF candidate depth = 20 (passed entirely to ssemble_context(), maintaining a max context chunk limit of 5 after deduplication).

### Methodology
Modified 
ag.py to skip slicing candidates[:5] for Hybrid mode.
Updated ssemble_context() to accept max_chunks=5 so it can iterate over the 20 candidates, deduplicate, and stop cleanly once the budget of 5 unique chunks is filled.
Executed the full 20-query evaluation suite via the API on the experimental configuration.

### Aggregate Results
* **Recall@3:** 0.3750 (Baseline: 0.3750) -> **No Change**
* **Recall@5:** 0.3750 (Baseline: 0.3750) -> **No Change**
* **MRR:** 0.2708 (Baseline: 0.2708) -> **No Change**
* **nDCG:** 0.2976 (Baseline: 0.2976) -> **No Change**

### Query-Level Findings
A focused query-by-query analysis running both Depth 5 and Depth 20 logic demonstrated exactly **0 differences** in the retrieved chunks and their ranks.
The expected relevant chunks were NOT sitting just below rank 5 waiting to be rescued by deduplication. 
The BM25 distractors that pushed the true answers down are distinctly different textual chunks (not duplicates), meaning they consume the 5 available context slots regardless of how many candidates are passed into the assembler.

### Latency Impact
* **Retrieval latency:** ~15.5 ms (Baseline: ~15.7 ms)
* **Context assembly:** Negligible impact.
* **Total latency:** Remained heavily dominated by LLM generation (~14s). Passing 20 chunks to the assembler added sub-millisecond overhead.

### Conclusion
**The hypothesis is not supported.** Increasing the pre-context candidate depth from 5 to 20 produces absolutely no improvement in retrieval quality without an intelligent relevance filter. Because ssemble_context() relies strictly on exact-text deduplication, it simply takes the first 5 unique chunks, which are the exact same 5 chunks obtained by truncating the list early. 

### Recommendation
**Do not adopt Depth 20.** I have reverted the candidate depth back to the baseline of 5.
For the next experiment, we should introduce an alpha weighting parameter to RRF (Experiment 2: Weighted RRF) to mathematically penalize BM25 noise *before* the chunks are sorted, ensuring the true semantic matches never drop below rank 5.

## Experiment 2 — Weighted Reciprocal Rank Fusion

### Hypothesis
The current RRF implementation applies equal weighting to Dense and Sparse retrieval rankings. Because BM25 keyword matching can introduce noisy distractors that push semantic matches down, assigning a greater weight to Dense retrieval (e.g., 0.7) should theoretically suppress BM25-induced noise and improve Hybrid retrieval quality.

### Method
Modified the 
eciprocal_rank_fusion algorithm to accept dense_weight and sparse_weight multipliers.
The mathematical implementation modifies standard RRF as follows:
WRRF(d) = α * (1 / (k + rank_dense(d))) + β * (1 / (k + rank_sparse(d)))
Where k=60.

### Configurations
* **Baseline:** α = 0.5, β = 0.5 (Mathematically equivalent to current default 1.0 / 1.0)
* **Experiment A:** α = 0.7, β = 0.3
* **Experiment B:** α = 0.8, β = 0.2

### Aggregate Metrics
| Configuration | Recall@3 | Recall@5 | MRR | nDCG |
| --- | --- | --- | --- | --- |
| Baseline (0.5/0.5) | 0.5000 | 0.5625 | 0.4500 | 0.4780 |
| Weighted (0.7/0.3) | 0.5000 | 0.5625 | 0.4396 | 0.4699 |
| Weighted (0.8/0.2) | 0.5000 | 0.5625 | 0.4427 | 0.4726 |

*Note: The isolated retrieval metrics evaluate purely context assembly hit-rates.*

### Query-Level Findings & Noise Analysis
The hypothesis completely inverted in practice. Increasing Dense weight and penalizing Sparse actually **degraded** ranking quality (MRR dropped from 0.4500 to 0.4396).
For example, on the query *"What are the main components of the LangChain ecosystem?"*:
* In the Baseline configuration, the correct chunk containing "LangSmith" achieved Rank 2.
* In the 0.7/0.3 configuration, this exact chunk was pushed down to Rank 3.
A raw examination of the independent retrieval pipelines revealed that for this specific query, Dense retrieval ranked the target chunk at Rank 5, while Sparse (BM25) provided complementary ranking signal that elevated it. Penalizing the sparse signal reduced the chunk's RRF score relative to purely semantic (but contextually wrong) distractors. 

### Latency
No measurable difference. Weighted arithmetic adds O(N) floating-point multiplications which execute in sub-millisecond time.

### Result
**REJECTED.**
Penalizing BM25 sparse retrieval does not selectively remove noise; it also suppresses critical exact-keyword signals that were successfully boosting correct chunks above Dense-only semantic distractors.

### Recommendation
Retain the modular weighting implementation in usion.py to allow future hyperparameter sweeps or query-adaptive routing, but **keep the production default weights at 1.0 / 1.0 (equal weighting)**. The current Hybrid/RRF balance is empirically superior to a Dense-heavy split for this dataset.

## Experiment 3: Semantic / Structural Chunking Optimization
**Status:** COMPLETED
**Objective:** Determine whether structural/semantic chunking improves retrieval quality over the baseline arbitrary fixed-length string chunking (size=1000, overlap=200).

### Hypothesis
The existing chunking strategy splits concepts across arbitrary character boundaries, degrading the semantic coherence of Dense embeddings and splitting exact-match context for BM25.

### Methodology
- Created an isolated semantic_chunk_text function in a temporary environment.
- Prioritized structural Markdown boundaries (\n# , \n## , \n\n, etc.) before falling back to character limits.
- Evaluated on the eval_dataset.json queries against the isolated Qdrant memory store and BM25 store.
- **Phase 3A:** Semantic chunking with overlap=0.
- **Phase 3B:** Semantic chunking with overlap=200.

### Results

| Metric | Base Hybrid | Base Dense | Base Sparse | Sem Hybrid (No Overlap) | Sem Dense (No Overlap) | Sem Sparse (No Overlap) | Sem Hybrid (Overlap 200) | Sem Dense (Overlap 200) | Sem Sparse (Overlap 200) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Recall@3** | 0.3125 | 0.3125 | 0.3125 | 0.3125 | 0.3125 | 0.2500 | **0.3125** | **0.3750** | 0.3125 |
| **Recall@5** | 0.3750 | 0.3750 | 0.3125 | 0.3750 | 0.3750 | 0.3125 | **0.3750** | **0.3750** | 0.3125 |
| **MRR**      | 0.2656 | 0.2552 | 0.2396 | 0.2448 | 0.2656 | 0.1479 | **0.2865** | **0.3333** | 0.2292 |
| **nDCG**     | 0.2933 | 0.2851 | 0.2582 | 0.2769 | 0.2933 | 0.1886 | **0.3082** | **0.3438** | 0.2500 |

### Findings
1. **Dense Retrieval Improved Significantly**: By preserving semantic boundaries, Dense MRR surged from  .2552 to  .3333 (+30% relative improvement) when using overlap=200. The embedding vectors now represent coherent concepts rather than fragments.
2. **Overlap is Mandatory for BM25**: When overlap=0 was tested, BM25 MRR collapsed from  .2396 to  .1479. Sparse retrieval relies heavily on term co-occurrence; hard boundaries without overlap split adjacent keywords, destroying exact-match signals.
3. **Hybrid Performance**: Overall Hybrid MRR improved from  .2656 to  .2865 with Semantic Chunking (Overlap 200). 

### Conclusion
**HYPOTHESIS CONFIRMED.** Semantic chunking improves dense retrieval quality. However, overlap must be maintained to preserve BM25 performance. 
We should adopt a semantic chunker with a 200-character overlap fallback.


## Experiment 4: Cross-Encoder Reranker Ablation
**Status:** COMPLETED
**Objective:** Determine whether the current Cross-Encoder reranker (cross-encoder/ms-marco-TinyBERT-L-2-v2) improves retrieval quality over the Hybrid RRF baseline without unacceptable latency.

### Hypothesis
Reranking the Top-20 fused candidates using a Cross-Encoder improves the final Top-5 context quality by establishing deeper semantic relationships between the query and text chunks, which RRF alone cannot achieve.

### Configuration
* **Control:** Dense + BM25 -> RRF (limit=20) -> truncate to Top 5
* **Experiment:** Dense + BM25 -> RRF (limit=20) -> TinyBERT Reranker -> truncate to Top 5
* **Frozen Variables:** Semantic chunking (overlap=200), all-MiniLM embeddings, BM25 tokenizer, RRF equal weights (1.0/1.0).
* **Dataset:** 16 answered queries from the standard evaluation suite.

### Methodology
Built an isolated retrieval-only ablation evaluator (
exus/scripts/run_experiment4.py) that bypassed LLM generation to purely measure candidate rank-shifting. The exact same 20 RRF candidates were passed to the reranker to perfectly isolate the reranking behavior.

### Results

| Metric | RRF (Control) | RRF + Reranker | Δ |
|---|---:|---:|---:|
| **Recall@3** | 0.4375 | 0.5000 | +0.0625 |
| **Recall@5** | 0.5000 | 0.5000 | 0.0000 |
| **MRR** | 0.4083 | 0.4687 | +0.0604 |
| **nDCG** | 0.4304 | 0.4769 | +0.0465 |

### Paired Analysis & Rank Shift
* **Improved (HELPED):** 2 queries
* **Unchanged:** 13 queries
* **Degraded (HURT):** 1 query
* **Rank Shift:** 2 relevant chunks promoted, 1 demoted. Average rank change: +1.67 positions.

### Candidate Coverage (Ceiling Analysis)
* **Relevant evidence present in RRF Top-20:** 8 queries (50%)
* **Relevant evidence missing from RRF Top-20:** 8 queries (50%)
* *Insight:* The reranker's impact is severely bottlenecked by initial retrieval. Half the time, the correct chunk isn't even in the candidate pool for the reranker to rescue.

### Query-Level Findings
* **Major Wins:** 
  * *"What are the main components of the LangChain ecosystem?"* - RRF ranked the answer at 3 and 5. TinyBERT recognized the semantic relationship and pushed them to 1 and 2 (MRR 0.33 -> 1.0).
  * *"What is a Document in LangChain?"* - RRF ranked it at 5. TinyBERT pushed it to 1 (MRR 0.20 -> 1.0).
* **Major Failures:**
  * *"Can I trace my LLM apps?"* - RRF correctly ranked the chunk at 1. The reranker demoted it slightly to rank 2. (MRR 1.0 -> 0.5).

### Latency
* **RRF Latency:** ~12.2 ms
* **Reranker Latency:** ~78.3 ms
* **Combined Overhead:** ~90.5 ms
* *Conclusion:* 78ms is an entirely acceptable overhead for a +15% relative improvement in MRR (0.408 -> 0.468).

### Conclusion
**SUPPORTED.** 
The current TinyBERT Cross-Encoder provides a measurable, descriptive improvement in retrieval quality (MRR +0.06, nDCG +0.04) with an acceptable sub-100ms latency cost. It successfully salvages semantic relationships that BM25/Dense RRF ranks poorly. 
However, the system's ceiling is strictly bound by upstream candidate coverage (50%).


## Experiment 5: Upstream Candidate Coverage
**Status:** INCONCLUSIVE (Corpus Bound)
**Objective:** Determine whether increasing the upstream candidate pool (Dense/Sparse/RRF K) from 20 to 40 (or 60) recovers relevant chunks that previously missed the reranker's candidate pool, thereby improving final Top-5 retrieval metrics.

### Hypothesis
A significant portion of relevant chunks are ranked below 20 by BM25/Dense retrieval. Increasing upstream retrieval depth to 40 will pull these chunks into the candidate pool, allowing the TinyBERT reranker to rescue them and promote them into the final Top-5 context.

### Configuration
* **Control:** K=20 (Dense=20, Sparse=20, RRF pool=20, Rerank Top 5)
* **Experiment 5A:** K=40 (Dense=40, Sparse=40, RRF pool=40, Rerank Top 5)
* **Experiment 5B:** K=60 (Dense=60, Sparse=60, RRF pool=60, Rerank Top 5)
* **Frozen Variables:** Semantic chunking (overlap=200), reranker model, test dataset.

### Candidate Coverage & Metrics Results

| Metric | K20 | K40 | K60 | Δ (K20 -> K40) |
|---|---:|---:|---:|---:|
| **Recall@3** | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| **Recall@5** | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| **MRR** | 0.4688 | 0.4688 | 0.4688 | 0.0000 |
| **nDCG** | 0.4769 | 0.4769 | 0.4769 | 0.0000 |

### Critical Finding: Corpus Saturation
**The experiment yielded absolutely zero metric changes.**
A forensic analysis of the dataset revealed exactly why: **N = 19**.
The entire production corpus (consisting of the 3 test documents) produces only 19 semantic chunks in total.
Therefore, retrieving K=20 already pulls 100% of the entire database. 
The 8 queries classified as "missing from the Top-20 candidate pool" are missing because the expected ground truth text (e.g., "langchain expression language", "machine learning") literally does not exist anywhere within the ingested documents.

### Latency
* **K20:** ~115ms (Retrieval 14ms + Rerank 101ms)
* **K40:** ~116ms (Retrieval 12ms + Rerank 103ms)
* **K60:** ~114ms (Retrieval 13ms + Rerank 101ms)
* *Note:* Latency remains identical because the reranker is only ever processing the same 19 physical chunks.

### Conclusion
**INCONCLUSIVE.** We cannot measure the effect of upstream candidate depth > 20 because the existing evaluation corpus contains fewer than 20 chunks. The current ceiling on retrieval (MRR ~0.468) is entirely artificial, bound by unanswerable evaluation queries rather than retrieval system failures. 

### Recommendation
Retain K=20 for production. Before conducting any further retrieval depth or threshold experiments, the evaluation dataset must be paired with a representative document corpus that actually contains the expected answers, and N (total chunks) must heavily exceed K (candidate depth).

## Experiment P1: Context Budget Ablation
**Status:** INCONCLUSIVE (Corpus Bound)
**Objective:** Determine whether Syntera's fixed final context limit of 5 creates a measurable quality ceiling now that document-aware context expansion exists.

### Hypothesis
Increasing final evidence budget improves answer quality for queries requiring multi-chunk or distributed evidence.

### Configuration
* **Budgets (K):** 3, 5, 8, 10, 15
* **Modes:** Neighbor Expansion OFF vs. Neighbor Expansion ON
* **Frozen Variables:** Semantic chunking, embeddings, BM25, RRF weights, Reranker Depth (20), LLM prompt.

### Results summary
* **Latency:** Generation latency scaled linearly with context budget. K=15 incurred a ~40% latency penalty compared to K=5 due to increased token processing. Retrieval latency remained stable.
* **Expansion effectiveness:** Expansion successfully transformed highly scored but fragmented chunks into logical reading blocks. It never crossed document boundaries. For K=5, expansion typically added 2-3 contextual chunks, resulting in an effective context size of ~8 chunks grouped logically.
* **Corpus Limitation:** The entire test corpus contains only 19 chunks across 3 physical documents. When K=15, the system selects ~79% of the entire database. When Expansion is enabled at K>8, the context window frequently contained the entire database.

### Conclusion
**INCONCLUSIVE / CORPUS-BOUND.**
While Neighbor Expansion clearly improves paragraph continuity, we cannot definitively adopt K>5 because the corpus is too small to measure the precision drop-off or distractor penalty that occurs in a real 10,000+ chunk database. 

### Recommendation
**RETAIN K=5.** (Combined with expand_neighbors=True, this provides an effective contextual footprint of ~7-9 chunks, which is highly optimal for the current architecture). Next phase should be P2 Table Extraction.

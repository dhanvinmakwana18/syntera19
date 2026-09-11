# CURRENT ARCHITECTURE REALITY AUDIT

## 1. Executive Summary
This audit strips away aspirational documentation and examines the concrete implementation of Syntera as of the latest commit (11acf58). While Syntera is marketed in its README.md as an "Autonomous Agentic RAG & Multi-Modal AI Engine", the codebase reveals a more grounded reality. Syntera is, at its core, a **static, high-quality RAG pipeline** featuring structural PDF ingestion, hybrid retrieval, and cross-encoder reranking. The "Agentic" and "Multi-Modal" features are either rudimentary LLM prompts or entirely missing.

## 2. Actual Architecture
The actual running architecture operates as follows:
`	ext
User Request
  ↓
FastAPI (Port 8001)
  ↓
api/router.py (LLM Prompt Classification)
  ├── DIRECT: Standard LLM Generation (Ollama / Qwen2.5)
  ├── AGENTIC: workflow.py (Another LLM Prompt -> Standard RAG)
  └── RAG: Standard RAG Pipeline
        ├── Dense Search (Qdrant via all-MiniLM-L6-v2)
        ├── Sparse Search (BM25)
        ├── Reciprocal Rank Fusion (RRF)
        ├── Reranker (TinyBERT Cross-Encoder)
        ├── Generation (Ollama / Qwen2.5)
        └── Grounding Check (LLM as a Judge)
`

## 3. Actual Query Flow
1. Query enters pi_router.post("/chat").
2. oute_query sends the query to the LLM asking it to classify into DIRECT, RAG, MULTI_MODAL, or AGENTIC.
3. The string is parsed, and if it matches "RAG", it calls etrieve_documents(query).
4. The chunks are fetched, fused, and reranked.
5. The LLM generates the answer.
6. alidate_citations runs a regex to ensure any [Source X] actually exists in the retrieved context.
7. evaluate_support prompts the LLM to output "YES/NO" on whether the claim is supported.
8. The JSON response is returned.

## 4. Actual Ingestion Flow
1. PyMuPDF extracts text blocks and table blocks.
2. Tables are preserved structurally using Markdown.
3. Text blocks are recursively chunked (1000 tokens, 200 overlap).
4. update_heading_stack extracts Markdown headers and assigns a section_path (e.g., Root > Header 1).
5. Vectors are encoded via SentenceTransformers and pushed to Qdrant alongside BM25 indexing.

## 5. Actual Retrieval Flow
1. 	ransform_query is called (but currently does minimal transformation).
2. Qdrant performs Dense Search (Candidate Depth K=20).
3. ank_bm25 performs Sparse Search (Candidate Depth K=20).
4. eciprocal_rank_fusion merges them using 1 / (60 + rank).
5. eranker_service scores the fused candidates using cross-encoder/ms-marco-TinyBERT-L-2-v2.
6. Top =5$ are returned.

## 6. Actual Generation Flow
1. The llm_provider routes to Ollama (localhost:11434).
2. If ConnectionError occurs, it natively falls back to a loaded HuggingFace Pipeline (Qwen/Qwen2.5-0.5B-Instruct on CPU).
3. The context and prompt are passed, generation is returned.

## 7. Component Inventory

| Component | Exists? | Actual Implementation | Status | Evidence |
|-----------|---------|-----------------------|--------|----------|
| PDF Parsing | YES | PyMuPDF structural extraction | Production | parser.py:6 |
| Chunking | YES | Recursive text split (1000tk/200) + Table preservation | Production | parser.py:64 |
| Embeddings | YES | ll-MiniLM-L6-v2 | Production | Qdrant Client Setup |
| Vector DB | YES | Qdrant (Local) | Production | qdrant_client.py |
| Sparse Retrieval| YES | ank_bm25 | Production | m25_store.py |
| Hybrid RRF | YES | standard RRF weighting | Production | usion.py:1 |
| Reranker | YES | ms-marco-TinyBERT | Production | eranker.py |
| Context Assembly| YES | Simple string concatenation + Neighbors | Production | ssembler.py |
| LLM Generation | YES | Ollama / HF Pipeline Fallback | Production | llm.py |
| Citation Gen | YES | Prompt-enforced [Source X] | Production | outer.py:108 |
| Grounding Check | YES | Regex Citation + LLM-as-a-judge Entailment | Production | grounding.py:26 |
| Query Routing | YES | LLM Prompt Classification | Experimental | gents/router.py:3 |
| Agentic Workflow| YES | Linear LLM intent re-evaluation | Experimental | workflow.py:19 |
| Multi-Modal | NO | Empty Directory | Planned | multimodal/ |

## 8. Router Audit
The README.md describes an "Agentic Router (Query Classification)" that deterministically routes queries. 
**Reality:** It is not deterministic. It is a single shot LLM prompt in services/agents/router.py. It is highly susceptible to latency bottlenecks (requires a full LLM pass before retrieval even begins) and hallucination (routing to MULTI_MODAL when no such pipeline exists).

## 9. Multimodal Audit
**Reality:** NOT PRESENT. The directory services/multimodal exists but is completely empty.

## 10. Grounding/Citation Audit
**Reality:** Syntera verifies two things:
1. **Citation Existence:** alidate_citations strips or warns if [Source 99] is used but Chunk 99 wasn't retrieved. 
2. **Semantic Entailment:** evaluate_support asks the LLM to output YES/NO if the generated answer is supported by the context.
**Crucial limitation:** Because the LLM acts as its own judge, it is highly vulnerable to Confirmation Bias.

## 11. Evaluation Audit
**Reality:** 
- The eval_dataset_v2.json contains 20 queries across a 19-chunk (3 PDF) corpus.
- This is a localized smoke-test, not a generalized evaluation.
- The most recent benchmark (FINAL_BENCHMARK.md) proves the architectural piping works (Hybrid + Reranking Recall@3 = 0.90) but is NOT proof of scale.

## 12. README vs Reality
- **Contradiction:** "Autonomous Agentic RAG". Syntera is not autonomous; it is a request-response API.
- **Contradiction:** "Agentic Routing". It is LLM prompt routing.
- **Contradiction:** "Multi-modal AI Engine". Code does not exist.
- **Accuracy:** The RAG claims (Qdrant, BM25, RRF, Cross-Encoder) are 100% accurate and mathematically sound in the codebase.

## 13. AGENTS.md vs Reality
- **Alignment:** Perfect. The AGENTS.md strictly bans treating Syntera as an "Autonomous AI agent" and rightfully classifies it as an experimental RAG system requiring baseline validation, which perfectly mirrors the actual Python implementation.

## 14. Missing/Planned Components
- NLI Entailment verification (to replace LLM-as-a-judge).
- XGBoost/LightGBM deterministic routing (to replace LLM latency routing).
- Dynamic abstention (Thresholding).
- Parent-Child structural chunking.

## 15. Technical Risks
- **Latency:** The fallback Qwen2.5-0.5B-Instruct model runs on CPU. Generating an answer takes ~15 seconds. Routing using this model doubles the latency to ~30 seconds because the LLM is called twice.
- **Token Limits:** Hardcoded K=5 truncation drops relevant information on complex multi-hop queries.

## 16. Recommended Next Research Area
**Fast-Path Deterministic Routing.** Replacing the LLM-based oute_query with a heuristic or lightweight ML classifier (FastText/LightGBM) will instantly cut system latency in half and prevent invalid routing states.

==================================================
CURRENT SYNTERA DEFINITION
==================================================
Syntera is a locally-hosted, high-precision RAG pipeline built on PyMuPDF structural ingestion, hybrid Qdrant/BM25 retrieval, and cross-encoder reranking, currently limited by LLM-dependent routing and evaluation bottlenecks.

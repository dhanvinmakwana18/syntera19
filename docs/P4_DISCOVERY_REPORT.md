# SYNTERA INTELLIGENCE CORE: P4 DISCOVERY REPORT

This document represents the final engineering discovery and architectural map for the transition from P3 (Deterministic RAG) to P4 (Adaptive Intelligence Core). 

## A. CURRENT ARCHITECTURE MAP
Based on an inspection of the repository (commit `752552b`):

**Frontend**: React 19 / Vite application that communicates with FastAPI.
**Backend**: FastAPI (`backend/run_server.py`) serving an AI Pipeline.
**Agentic Routing (`backend/services/agents/router.py`)**: A single, static LLM prompt that forces the query into one of four rigid categories (`DIRECT`, `RAG`, `MULTI_MODAL`, `AGENTIC`).
**Execution Workflow (`backend/services/agents/workflow.py`)**: A deterministic script that checks the router string and hardcodes `retrieve_documents` if "RAG" is found.
**Retrieval Engine (`backend/services/retrieval/rag.py`)**: A robust pipeline offering `dense`, `sparse`, `hybrid`, and `rerank` modes using Qdrant + BM25, Reciprocal Rank Fusion, and Cross-Encoder (TinyBERT) reranking.
**Context Assembly (`backend/services/rag/assembler.py`)**: Supports document-aware neighbor expansion and deduplication.
**Generation**: The context is stuffed into a static prompt and generation falls back to `Qwen/Qwen2.5-0.5B-Instruct` if Ollama is unavailable.

## B. CURRENT CAPABILITY MAP
**What Syntera CAN do today**:
- Hybrid Document Retrieval (Dense + BM25).
- Structure-Aware extraction (PDF tables, blocks, section propagation).
- Reciprocal Rank Fusion & Cross-Encoder reranking.
- Neighbor chunk expansion (restoring semantic continuity).
- Deterministic routing to RAG or DIRECT answering.
- Providing full observability traces of backend execution latencies and chunk scores.
- Passing a 20-query rigorous engineering benchmark (21/21 tests pass).

**What Syntera is NOT**:
- It is **not** an agent. It cannot plan, it cannot decompose complex tasks, and it cannot react to failure dynamically. 
- It is **not** capable of self-verification.
- It is **not** a data science analyst. It has no tools for executing code, running SQL, or visualizing data.
- It does **not** have memory beyond appending to a static string limit.

## C. FAILURE MAP
1. **The Routing Fallacy**: `router.py` uses an LLM to pick 1 of 4 rigid strings. This fails on complex tasks (e.g. "Compare the sales from X and Y") because the router has no capability to decompose the task or retrieve twice.
2. **Deterministic Evidence Blindness**: The system pulls $K$ chunks regardless of whether the query needs 1 sentence or an entire book. If the evidence is insufficient, it relies entirely on the local LLM to follow the instruction "If you cannot find the answer...". Small LLMs (like Qwen2.5-0.5B fallback) fail this routinely and hallucinate.
3. **Citation Hallucinations**: Grounding currently checks for lexical overlap rather than semantic entailment, leading to false positives on citations.
4. **Tool Isolation**: There is no mechanism to decide if a query needs *both* RAG and Python execution. It's either/or.

## D. RESEARCH LANDSCAPE
- **Task Understanding & Routing**: 
  - *LLM-as-Router*: High latency, easy to implement.
  - *Semantic Router (Embedding thresholding)*: Low latency, highly robust for known intents.
- **Query Complexity & Planning**:
  - *ReAct / Toolformer*: Standard agent loops. Too slow for simple queries.
  - *Plan-and-Solve (Self-Discover)*: Decomposes the task first, then executes. Excellent for complex multi-hop.
- **Evidence Intelligence**:
  - *Self-RAG (Asai et al., 2023)*: Uses a fine-tuned model or specific critic prompts to output critique tokens (e.g., `[Relevant]`, `[Fully Supported]`).
  - *NLI (Natural Language Inference) Models*: Small, extremely fast cross-encoders (e.g., `DeBERTa-v3`) used strictly to verify if Context logically entails the generated Claim.
- **Tool Selection & Data Science**:
  - *Code Execution Agents (OpenAI Code Interpreter style)*: Generating and executing Python in a sandbox. Vital for data science tasks.

## E. DIFFERENTIATION OPPORTUNITIES
1. **Dynamic Adaptive RAG**: Syntera can be the first local-first system that calculates query complexity *before* retrieval. If a query is simple, it skips the expensive cross-encoder. If complex, it triggers iterative multi-hop retrieval.
2. **NLI-Powered Self-Correction**: Instead of relying on a weak generator LLM to refuse unanswerable questions, we use a dedicated NLI model to cross-reference the generated answer against the context. If entailment fails, Syntera dynamically retrieves more or formally refuses.
3. **Data Science / Python Sandbox**: Adding an autonomous Python execution engine that allows Syntera to ingest a dataset (CSV/Dataframe), write Pandas code to profile it, form hypotheses, and generate statistical charts.
4. **Transparent Planning**: Showing the user exactly *why* Syntera chose a tool, mirroring the backend trace observability we already built.

## F. RECOMMENDED ARCHITECTURE (SYNTERA INTELLIGENCE CORE)
We propose replacing the deterministic pipeline with a 4-Stage Intelligence Core:

1. **STAGE 1: FAST INTENT & COMPLEXITY ROUTER**
   - Use a lightweight fast embedding classifier (Semantic Router) or a small dedicated classifier LLM.
   - Output: Route (e.g., DataScience, RAG, Chat) + Complexity Score (0.0 to 1.0).

2. **STAGE 2: PLANNER & TOOL DISPATCHER**
   - If Complexity > Threshold: Engage a ReAct-style Planning Agent to break down the task into sub-queries.
   - If Complexity < Threshold: Route directly to the appropriate Tool Agent (e.g., RAG Tool, Python Tool).

3. **STAGE 3: THE KNOWLEDGE & EVIDENCE ENGINE (Evolution of RAG)**
   - RAG is now a *Tool* available to the Planner.
   - The RAG tool can be called multiple times iteratively (Multi-hop).

4. **STAGE 4: NLI VERIFIER**
   - A dedicated verification step. Before returning the final answer, an NLI model (e.g., `cross-encoder/nli-deberta-v3-small`) checks: `Premise: [Retrieved Context] -> Hypothesis: [Generated Answer]`.
   - If the NLI model returns `Contradiction` or `Neutral`, the system loops back to Stage 3 or refuses to answer.

## G. ALGORITHM SHORTLIST
- **Router**: Semantic Router (cosine similarity of query against prototypical intent embeddings).
- **Planner**: ReAct or Plan-and-Solve with constrained JSON schemas.
- **Verifier**: `cross-encoder/nli-deberta-v3-small` for claim-level entailment.
- **Data Science Engine**: Local Jupyter Kernel / Python REPL (e.g., `jupyter-client` or restricted `subprocess` execution) to allow Pandas/Matplotlib execution.

## H. EXPERIMENT PLAN
1. **Hypothesis 1 (NLI Verifier)**: A dedicated NLI cross-encoder can detect hallucinations in the fallback LLM with >90% accuracy compared to the LLM self-correcting.
   - *Test*: Pass the 20-query benchmark generation outputs through DeBERTa-NLI. Measure if it correctly flags the 3 known hallucination failures.
2. **Hypothesis 2 (Semantic Routing)**: Embedding-based intent classification is >5x faster and just as accurate as the current LLM router.
   - *Test*: Build a prototypical embedding index for "DIRECT", "RAG", "DATA_SCIENCE". Route 50 test queries.
3. **Hypothesis 3 (Python Execution)**: Syntera can successfully run generated Pandas code to answer a statistical question about a CSV.
   - *Test*: Implement a basic Python REPL tool. Pass a CSV and ask "What is the average X?".

## I. IMPLEMENTATION ROADMAP
- **P4.1 (Verification & Evidence)**: Implement the NLI Verifier stage. Eradicate hallucinated citations from the existing baseline.
- **P4.2 (Adaptive Routing)**: Replace `router.py` with the Semantic Intent & Complexity Router.
- **P4.3 (Data Science Tool)**: Implement a secure Python REPL tool capable of passing state (DataFrames) and returning console outputs/images.
- **P4.4 (Agentic Planner Loop)**: Implement the multi-step reasoning loop (Plan -> Execute Tool -> Observe -> Verify -> Finalize).

## J. WHAT NOT TO BUILD
- **Knowledge Graphs (Neo4j, etc.)**: High engineering overhead with minimal proven benefit over our existing structured hierarchical Qdrant + neighbor expansion approach.
- **Training a custom LLM**: Not cost-effective. We will rely on composable smaller models (cross-encoders, NLI) combined with off-the-shelf generative models.
- **Endless Tool Wrapping**: We will not add weather APIs, web search, or random calculators. We will add *only* two primary tools: Advanced RAG and Data Science Execution.

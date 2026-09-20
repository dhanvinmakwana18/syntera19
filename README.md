# Syntera

Autonomous Agentic RAG & Multi-Modal AI Engine

## 1. Overview
Syntera is an autonomous AI engine that combines agentic routing, retrieval-augmented generation (RAG), vector search, document intelligence, and multi-modal reasoning into a unified execution pipeline. It is built as a robust, local-first backend paired with a futuristic observability frontend.

## 2. Why Syntera
Unlike standard wrappers around commercial APIs, Syntera is designed to demonstrate real AI engineering principles:
- **Local-first**: Operates on local models via Ollama and SentenceTransformers.
- **Agentic Routing**: Deterministically classifies intents and executes the appropriate sub-system.
- **Traceability**: Exposes operational execution events (latency, tool calls, retrieval counts) safely to the user.

## 3. Architecture

```text
USER
 ↓
React/Vite Frontend
 ↓
FastAPI / Python API (Port 8001)
 ↓
Agentic Router (Query Classification)
 ├── DIRECT (LLM Generation)
 ├── RAG (Hybrid Search + Reranking)
 ├── MULTI_MODAL (Vision - Planned)
 └── AGENTIC (Complex Reasoning Workflow)
 ↓
Context Assembly (Deduplication & Grounding)
 ↓
Final Answer + Sources + Observability Trace
```

## 4. Core Features
- **Advanced RAG**: True Hybrid Search combining Dense (Qdrant) and Sparse (BM25) retrieval, fused via Reciprocal Rank Fusion (RRF), and refined with a Cross-Encoder Reranker (`ms-marco-TinyBERT`).
- **Grounding & Citations**: Explicit source preservation and citation verification. Output includes `grounded: true/false` status.
- **Agentic Workflow**: A routing system that chooses between tools (RAG, Direct LLM, Vision).
- **Execution Trace**: Every query generates a detailed observability trace outlining exact backend operations, latency, and routing decisions.
- **Structured Health Monitoring**: Comprehensive `/health` endpoint tracking the status of LLM, Vector Store, Embeddings, and Reranking components.

## 5. Tech Stack
- **Backend**: Python, FastAPI, Pydantic, Qdrant (Persistent), SentenceTransformers, Cross-Encoders, PyMuPDF, rank_bm25.
- **Frontend**: React 19, Vite, Tailwind CSS, Framer Motion, Lucide Icons.

## 6. Installation & Running Locally

1. **Install Backend Dependencies**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Or .\venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```

2. **Start Backend Server**:
   ```bash
   python run_server.py
   ```
   *Runs on port 8001. Ensure Ollama is running locally with the `llama3` model.*

3. **Install Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

4. **Start Frontend Server**:
   ```bash
   npm run dev
   ```

## 7. RAG Evaluation & Metrics

We conducted an end-to-end evaluation using a synthetic 10-query dataset covering factual, semantic, technical, and insufficient-evidence queries against indexed chunks (including the LangChain architectural design). 

### Methodology
- **Baselines Tested:** Dense (Qdrant), Sparse (BM25), Hybrid (RRF), Hybrid + Cross-Encoder Reranking.
- **Metrics:** Recall@3 (does the top-3 context contain the answer), MRR (Mean Reciprocal Rank), End-to-End Latency, and Grounded% (does the system correctly handle insufficient evidence and output citations).

### Evaluation Results
*Measurements over a 10-query validation dataset run locally.*

| Configuration | Recall@3 | MRR | Latency (ms) | Grounded/Cited % |
|---------------|----------|-----|--------------|------------------|
| **Dense** (Qdrant) | 0.375 | 0.3125 | ~6105ms | 0.0%* |
| **Sparse** (BM25) | 0.375 | 0.375 | ~6072ms | 0.0%* |
| **Hybrid** (RRF) | 0.375 | 0.3125 | ~9226ms | 0.0%* |
| **Hybrid + Rerank** | 0.375 | 0.375 | ~6202ms | 0.0%* |

*\* Note on Grounding: Since the local evaluation was performed utilizing the mock LLM fallback (due to Ollama absence in the pipeline test environment), the mock response does not output genuine `[Source X]` citations. Therefore, the citation validation step correctly evaluates to 0.0% cited. The system appropriately handled 100% of the insufficient evidence testing cases.*

## 8. Limitations & Future Improvements
- **Local Qdrant Test Lock**: Fixed by adding an in-memory test isolation mode (`TESTING=True`).
- **Grounding Limitation**: Currently, grounding primarily checks for the *existence* of citations (`[Source X]`) matched to retrieved chunks (lexical verification). True semantic entailment verification (checking if the cited claim matches the source) is skipped to optimize latency, but traces are clearly marked as `CITED` rather than `GROUNDED` to reflect this.
- **Small Evaluation Dataset**: The current eval dataset is small and optimized for smoke testing the pipeline structure.
- **Implemented**: Advanced Hybrid RAG, Cross-Encoder Reranking, Direct Generation, Document Ingestion, Qdrant + BM25 Stores, Agentic Routing, Observability Traces.
- **Planned**: Advanced Multi-Modal execution (Llava integration).


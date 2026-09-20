# Syntera Modular RAG: Domain Contracts (Phase 2)

## 1. Contract Philosophy
Phase 2 establishes domain boundaries using abstract base classes (ABCs) and domain-specific data models. True Modular RAG relies on contracts, not concrete implementations.
* High-level orchestration depends only on abstractions.
* Infrastructure (Qdrant, BM25, CrossEncoders) lives behind these boundaries.
* Pydantic is used for strict typing of data payloads crossing module boundaries.

## 2. Domain Models (`core/domain.py`)
All retrieval operations now communicate using pure domain models rather than arbitrary dictionaries or third-party schema formats.

*   `Node`: Represents a chunk of retrieved text, along with its unique ID, arbitrary metadata, and base score.
*   `RetrievalResult`: A wrapper around `Node` that attaches an operation-specific score (e.g. vector similarity, BM25 score, or RRF rank).
*   `Query`: Represents the user's intent. Currently just a text string, but typed to allow future additions like query-time filters or modalities.

## 3. Retriever Contract
**Location**: `core/contracts.py :: BaseRetriever`
```python
def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]: ...
```
*   Takes a generic `Query` and returns `List[RetrievalResult]`.
*   A retriever is solely responsible for sourcing candidates. It does NOT generate final answers, nor does it embed strings unless composed with an EmbeddingProvider.

## 4. FusionStrategy Contract
**Location**: `core/contracts.py :: BaseFusionStrategy`
```python
def fuse(self, candidate_lists: List[List[RetrievalResult]], limit: int = 5, **kwargs) -> List[RetrievalResult]: ...
```
*   Fuses `N` lists of `RetrievalResult` into a single ranked `List[RetrievalResult]`.
*   Abstracts away the "Dense + Sparse" assumption. Any number of candidate sources can be merged (e.g. Dense, Sparse, Knowledge Graph, Web Search).

## 5. Reranker Contract
**Location**: `core/contracts.py :: BaseReranker`
```python
def rerank(self, query: Query, candidates: List[RetrievalResult], limit: int = 5) -> List[RetrievalResult]: ...
```
*   Takes candidates and a query, computes cross-attention or prompt-based relevance, and returns a re-sorted and truncated list.

## 6. EmbeddingProvider Contract
**Location**: `core/contracts.py :: BaseEmbeddingProvider`
```python
def embed_text(self, text: str) -> List[float]: ...
def embed_texts(self, texts: List[str]) -> List[List[float]]: ...
```
*   Separates the generation of vector embeddings from the storage system. Can be backed by local models or remote APIs (OpenAI, Cohere).

## 7. VectorStore Contract
**Location**: `core/contracts.py :: BaseVectorStore`
```python
def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]: ...
def add_points(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]: ...
```
*   A pure persistence and approximate nearest neighbor (ANN) boundary.
*   **Crucial Rule**: The `BaseVectorStore` does NOT take strings and secretely embed them. It only accepts and returns vectors and metadata.

## 8. Dependency Direction
```
Orchestration (Pipeline)
       │
       ▼
[ Domain Contracts (core/contracts.py) ]
       │
       ├───► BaseRetriever
       ├───► BaseFusionStrategy
       ├───► BaseReranker
       ├───► BaseEmbeddingProvider
       └───► BaseVectorStore
       
[ Implementations ]
DenseRetriever -----> BaseEmbeddingProvider
               -----> BaseVectorStore
               
BM25Retriever ------> (Internal State)
```

## 9. Current Adapters & Implementations
To avoid breaking the system, Phase 2 implements these contracts alongside legacy compatibility adapters:

1.  **Embeddings**: `EmbeddingProvider(BaseEmbeddingProvider)` using SentenceTransformers.
2.  **Dense Retrieval**: `QdrantVectorStore(BaseVectorStore)` for pure vector operations. `DenseRetriever(BaseRetriever)` composes the embedding provider and the vector store.
3.  **Sparse Retrieval**: `BM25Retriever(BaseRetriever)` handles scoring.
4.  **Fusion**: `RRFFusionStrategy(BaseFusionStrategy)` provides domain-compliant Reciprocal Rank Fusion.
5.  **Reranking**: `CrossEncoderReranker(BaseReranker)` provides local reranking.

## 10. Compatibility Strategy
The existing `pipeline.py` currently expects singletons (`vector_store`, `bm25_store`, `reranker_service`) and raw dictionaries.
We have subclassed the new implementations to expose the legacy dictionary-based API (e.g. `VectorStore(DenseRetriever)` returns dicts for `search()`). This ensures the existing pipeline and API routes remain fully functional while the contracts are introduced.

## 11. Known Remaining Architectural Debt
*   **Assembler DB I/O**: `assembler.py` still queries Qdrant directly to expand chunk context.
*   **BM25 Sync**: `BM25Store.sync_from_qdrant` still iterates over Qdrant to build its index at startup.
*   **Pipeline God Object**: `pipeline.py` is still a hardcoded procedural script, merely calling legacy adapter methods.
*   **Global Singletons**: State is still held in global instances instead of being injected via an application container.

## 12. What Phase 3 Should Address
Phase 3 must orchestrate the pipeline rewrite:
1.  **Dependency Injection**: Remove global singletons. Instantiate contracts dynamically based on config.
2.  **Pipeline Refactor**: Rewrite `retrieve_documents()` to accept generic lists of `BaseRetriever`s and dynamically route/fuse them, rather than hardcoding `dense` and `sparse`.
3.  **Context Expansion**: Move neighbor expansion out of the formatter and into a `NodePostProcessor` contract.
4.  **Independent Ingestion**: Refactor data ingestion to feed Qdrant and BM25 directly, allowing BM25 to drop its dependency on Qdrant synchronization.

import pytest
from typing import List
from core.domain import Query, RetrievalResult, Node
from retrieval.engine import RetrievalPipeline
from retrieval.assembler import PipelineContextAssembler
from tests.backend.test_contracts import FakeRetriever, FakeReranker, FakeVectorStore
from core.contracts import BaseFusionStrategy

class FakeFusionStrategy(BaseFusionStrategy):
    def fuse(self, candidate_lists: List[List[RetrievalResult]], limit: int = 5, **kwargs) -> List[RetrievalResult]:
        fused = [c for lst in candidate_lists for c in lst]
        return fused[:limit]

class ExceptionRetriever(FakeRetriever):
    def retrieve(self, query: Query, limit: int = 5, **kwargs):
        raise RuntimeError("Retriever explicitly failed")

def test_pipeline_with_one_retriever():
    retriever = FakeRetriever()
    pipeline = RetrievalPipeline(retrievers=[retriever])
    result = pipeline.run(Query(text="hello"))
    
    assert len(result.candidates) == 2
    assert result.trace[0]["stage"] == "retrieval_0"
    assert result.trace[0]["status"] == "success"

def test_pipeline_with_multiple_retrievers():
    retriever1 = FakeRetriever()
    retriever2 = FakeRetriever()
    fusion = FakeFusionStrategy()
    pipeline = RetrievalPipeline(retrievers=[retriever1, retriever2], fusion_strategy=fusion)
    
    result = pipeline.run(Query(text="hello"))
    assert len(result.candidates) > 0
    
    stages = [t["stage"] for t in result.trace]
    assert "retrieval_0" in stages
    assert "retrieval_1" in stages
    assert "fusion" in stages

def test_pipeline_replace_fusion():
    retriever1 = FakeRetriever()
    retriever2 = FakeRetriever()
    fusion = FakeFusionStrategy()
    pipeline = RetrievalPipeline(retrievers=[retriever1, retriever2], fusion_strategy=fusion)
    result = pipeline.run(Query(text="test"))
    
    assert [t for t in result.trace if t["stage"] == "fusion"][0]["strategy"] == "FakeFusionStrategy"

def test_pipeline_replace_reranker():
    pipeline = RetrievalPipeline(
        retrievers=[FakeRetriever()],
        reranker=FakeReranker()
    )
    result = pipeline.run(Query(text="test"))
    assert [t for t in result.trace if t["stage"] == "reranking"][0]["reranker"] == "FakeReranker"

def test_retriever_failure_isolation():
    failing_retriever = ExceptionRetriever()
    working_retriever = FakeRetriever()
    
    # We still need a fusion strategy to merge lists. If fusion strategy expects 2 lists and one fails,
    # the engine only passes the successful ones. 
    pipeline = RetrievalPipeline(
        retrievers=[failing_retriever, working_retriever]
    )
    result = pipeline.run(Query(text="test"))
    
    # The pipeline should still return candidates from the working retriever
    assert len(result.candidates) == 2
    
    # Check trace
    fail_trace = [t for t in result.trace if t["stage"] == "retrieval_0"][0]
    assert fail_trace["status"] == "failure"
    assert "Retriever explicitly failed" in fail_trace["error"]
    
    success_trace = [t for t in result.trace if t["stage"] == "retrieval_1"][0]
    assert success_trace["status"] == "success"

def test_pipeline_no_global_instantiation():
    import sys
    # Verify we can run a pipeline purely with fakes without importing heavy infra
    pipeline = RetrievalPipeline(retrievers=[FakeRetriever()])
    result = pipeline.run(Query(text="test"))
    assert len(result.candidates) > 0

def test_vector_store_receives_vectors():
    store = FakeVectorStore()
    results = store.search(query_vector=[0.1, 0.2, 0.3])
    assert len(results) == 1

def test_bm25_independent_from_qdrant():
    from vectorstore.bm25_store import BM25Store
    import os
    if os.path.exists("test_bm25.pkl"):
        os.remove("test_bm25.pkl")
    store = BM25Store(persist_path="test_bm25.pkl")
    # Add texts doesn't use Qdrant
    store.add_texts(["hello world", "other doc", "third doc"], [{"source": "test"}, {"source": "test2"}, {"source": "test3"}], ["1", "2", "3"])
    
    # Reload from disk, Qdrant not involved
    store2 = BM25Store(persist_path="test_bm25.pkl")
    print(store2.corpus)
    print(store2.bm25)
    results = store2.search("hello")
    assert len(results) == 1
    assert results[0]["id"] == "1"
    
    if os.path.exists("test_bm25.pkl"):
        os.remove("test_bm25.pkl")

def test_context_assembly_no_qdrant():
    builder = PipelineContextAssembler()
    candidates = [
        RetrievalResult(node=Node(id="1", text="chunk 1", metadata={"source": "A", "chunk_index": 1}), score=1.0)
    ]
    result = builder.assemble(candidates)
    context_str = result.text
    sources = result.sources
    assert "chunk 1" in context_str
    assert len(sources) == 1

from core.contracts import BaseRetriever, BaseVectorStore, BaseEmbeddingProvider, BaseReranker
from core.domain import Query, RetrievalResult, Node
from typing import List, Dict, Any, Optional

class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def embed_text(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
        
    @property
    def vector_size(self) -> int:
        return 3

class FakeVectorStore(BaseVectorStore):
    def __init__(self):
        self.points = []
    
    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        return [RetrievalResult(node=Node(id="fake_id", text="fake vector store match"), score=0.99)]
        
    def add_points(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        return ["fake_id"]
        
    def get_points_by_ids(self, ids: List[str]) -> List[Node]:
        return [Node(id="fake_id", text="fake vector store match")]

class FakeRetriever(BaseRetriever):
    def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        return [
            RetrievalResult(node=Node(id="doc1", text="match 1"), score=0.9),
            RetrievalResult(node=Node(id="doc2", text="match 2"), score=0.8)
        ]

class FakeReranker(BaseReranker):
    def rerank(self, query: Query, candidates: List[RetrievalResult], limit: int = 5) -> List[RetrievalResult]:
        # Just reverse the order and modify scores for test
        candidates_copy = candidates[:]
        candidates_copy.reverse()
        for idx, c in enumerate(candidates_copy):
            c.score = float(len(candidates_copy) - idx)
        return candidates_copy[:limit]

def test_fake_retriever():
    retriever = FakeRetriever()
    results = retriever.retrieve(Query(text="test query"))
    assert len(results) == 2
    assert results[0].node.id == "doc1"
    
def test_fake_embedding_provider():
    provider = FakeEmbeddingProvider()
    vec = provider.embed_text("hello")
    assert len(vec) == 3

def test_fake_vector_store():
    store = FakeVectorStore()
    results = store.search([0.1, 0.2, 0.3])
    assert len(results) == 1
    assert results[0].node.id == "fake_id"

def test_fake_reranker():
    reranker = FakeReranker()
    cands = [
        RetrievalResult(node=Node(id="1", text="a"), score=1.0),
        RetrievalResult(node=Node(id="2", text="b"), score=0.5)
    ]
    res = reranker.rerank(Query(text="q"), cands)
    assert res[0].node.id == "2"
    assert res[1].node.id == "1"

def test_rrf_fusion_with_fake_retrievers():
    from retrieval.fusion import RRFFusionStrategy
    
    retriever1 = FakeRetriever()
    retriever2 = FakeRetriever()
    
    q = Query(text="test query")
    list1 = retriever1.retrieve(q)
    list2 = retriever2.retrieve(q)
    
    fusion = RRFFusionStrategy()
    results = fusion.fuse([list1, list2])
    
    # doc1 is rank 0 in both lists. score = 2 * (1 / (60 + 0 + 1)) = 2/61
    # doc2 is rank 1 in both lists. score = 2 * (1 / (60 + 1 + 1)) = 2/62
    assert len(results) == 2
    assert results[0].node.id == "doc1"
    assert results[1].node.id == "doc2"

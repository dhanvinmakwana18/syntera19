from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from core.config import settings
from core.contracts import BaseVectorStore, BaseRetriever, BaseEmbeddingProvider
from core.domain import Query, RetrievalResult, Node
from typing import List, Dict, Any, Optional
import uuid
import os

class QdrantVectorStore(BaseVectorStore):
    def __init__(self, vector_size: int):
        if settings.QDRANT_STORAGE_PATH == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_points(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
            
        points = [
            PointStruct(id=doc_id, vector=emb, payload=payload)
            for emb, payload, doc_id in zip(vectors, payloads, ids)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return ids

    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        results = []
        for hit in search_result.points:
            payload = hit.payload or {}
            text = payload.get("text", "")
            node = Node(id=str(hit.id), text=text, metadata={k: v for k, v in payload.items() if k != "text"}, score=hit.score)
            results.append(RetrievalResult(node=node, score=hit.score))
        return results

    def get_points_by_ids(self, ids: List[str]) -> List[Node]:
        if not ids:
            return []
        try:
            points = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_payload=True, with_vectors=False)
            nodes = []
            for p in points:
                payload = p.payload or {}
                text = payload.get("text", "")
                nodes.append(Node(id=str(p.id), text=text, metadata={k: v for k, v in payload.items() if k != "text"}))
            return nodes
        except Exception:
            return []


class DenseRetriever(BaseRetriever):
    def __init__(self, vector_store: BaseVectorStore, embedding_provider: BaseEmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        query_vector = self.embedding_provider.embed_text(query.text)
        return self.vector_store.search(query_vector=query_vector, limit=limit)





def create_qdrant_retriever(embedding_provider: BaseEmbeddingProvider) -> BaseRetriever:

    vector_store = QdrantVectorStore(vector_size=embedding_provider.vector_size)

    return DenseRetriever(vector_store=vector_store, embedding_provider=embedding_provider)




def register(registry):
    registry.register_retriever('qdrant', create_qdrant_retriever)

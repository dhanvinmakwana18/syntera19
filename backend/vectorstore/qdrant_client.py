from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from core.config import settings
from providers.embeddings import embedding_provider
import uuid
import os

class VectorStore:
    def __init__(self):
        if settings.QDRANT_STORAGE_PATH == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_provider.vector_size, distance=Distance.COSINE),
            )

    def add_texts(self, texts: list[str], metadatas: list[dict], ids: list[str] = None):
        embeddings = embedding_provider.embed_texts(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
            
        points = [
            PointStruct(
                id=doc_id,
                vector=emb,
                payload={"text": text, **meta}
            )
            for text, emb, meta, doc_id in zip(texts, embeddings, metadatas, ids)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return ids

    def search(self, query: str, limit: int = 5, filters: dict = None):
        query_vector = embedding_provider.embed_text(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return [{"score": hit.score, "payload": hit.payload, "id": hit.id} for hit in search_result.points]

    def get_points_by_ids(self, ids: list[str]):
        if not ids:
            return []
        try:
            points = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_payload=True, with_vectors=False)
            return [{"payload": p.payload, "id": str(p.id)} for p in points]
        except Exception:
            return []

vector_store = VectorStore()

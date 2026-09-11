import os
import sys

sys.path.insert(0, os.path.abspath('syntera/backend'))

from vectorstore.qdrant_client import vector_store
from vectorstore.bm25_store import bm25_store
from ingestion.parser import ingest_document

def migrate():
    print("MIGRATING TO V2 DOCUMENT-AWARE SCHEMA...")
    
    # 1. Clear existing collection
    if vector_store.client.collection_exists(vector_store.collection_name):
        vector_store.client.delete_collection(vector_store.collection_name)
        print("Deleted old Qdrant collection.")
        
    # Recreate it
    from qdrant_client.models import VectorParams, Distance
    from providers.embeddings import embedding_provider
    vector_store.client.create_collection(
        collection_name=vector_store.collection_name,
        vectors_config=VectorParams(size=embedding_provider.vector_size, distance=Distance.COSINE),
    )
    
    # Clear BM25
    bm25_store.corpus = []
    bm25_store.doc_ids = []
    bm25_store.metadatas = []
    bm25_store.bm25 = None
    bm25_store._is_synced = False
    
    # 2. Re-ingest documents
    docs_dir = os.path.abspath('syntera/data/documents')
    for file in os.listdir(docs_dir):
        if file.endswith('.txt') or file.endswith('.md') or file.endswith('.pdf'):
            path = os.path.join(docs_dir, file)
            print(f"Ingesting {file}...")
            chunks = ingest_document(path, file)
            print(f" -> {chunks} chunks ingested with V2 metadata.")

    print("Migration complete. Vectors now have hierarchical section metadata and deterministic UUIDs.")

if __name__ == '__main__':
    migrate()

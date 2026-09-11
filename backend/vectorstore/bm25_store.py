import re
from rank_bm25 import BM25Okapi
import uuid

def tokenize(text: str):
    """Basic lightweight tokenizer for BM25."""
    # Convert to lowercase and split on non-alphanumeric characters
    return [word for word in re.split(r'\W+', text.lower()) if word]

class BM25Store:
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.metadatas = []
        self.bm25 = None
        self._is_synced = False

    def sync_from_qdrant(self, vector_store):
        """Synchronize BM25 index from Qdrant on startup."""
        print("Synchronizing BM25 index from Qdrant...")
        self.corpus = []
        self.doc_ids = []
        self.metadatas = []
        
        try:
            records, next_page_offset = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            if records:
                for r in records:
                    if r.payload and "text" in r.payload:
                        text = r.payload["text"]
                        meta = {k: v for k, v in r.payload.items() if k != "text"}
                        self.corpus.append(text)
                        self.metadatas.append(meta)
                        self.doc_ids.append(str(r.id))
                        
            if self.corpus:
                tokenized_corpus = [tokenize(doc) for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"BM25 index synchronized with {len(self.corpus)} chunks.")
            else:
                self.bm25 = None
                print("BM25 index empty (no chunks found).")
                
            self._is_synced = True
        except Exception as e:
            print(f"Failed to sync BM25 index from Qdrant: {e}")

    def add_texts(self, texts: list[str], metadatas: list[dict], doc_ids: list[str]):
        """Add new texts to the BM25 index."""
        if not texts:
            return
            
        self.corpus.extend(texts)
        self.metadatas.extend(metadatas)
        self.doc_ids.extend(doc_ids)
        
        # Re-initialize the entire BM25 index (fine for MVP scale)
        tokenized_corpus = [tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, limit: int = 5):
        """Search the BM25 index."""
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top scores > 0
        top_indices = [i for i, score in enumerate(doc_scores) if score > 0]
        top_indices = sorted(top_indices, key=lambda i: doc_scores[i], reverse=True)[:limit]
        
        results = []
        for i in top_indices:
            results.append({
                "score": float(doc_scores[i]),
                "payload": {"text": self.corpus[i], **self.metadatas[i]},
                "id": self.doc_ids[i]
            })
            
        return results

bm25_store = BM25Store()

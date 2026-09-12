import re
from rank_bm25 import BM25Okapi
import uuid
from core.contracts import BaseRetriever
from core.domain import Query, RetrievalResult, Node
from typing import List

def tokenize(text: str):
    """Basic lightweight tokenizer for BM25."""
    # Convert to lowercase and split on non-alphanumeric characters
    return [word for word in re.split(r'\W+', text.lower()) if word]

class BM25Retriever(BaseRetriever):
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.metadatas = []
        self.bm25 = None
        self._is_synced = False

    def retrieve(self, query: Query, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = tokenize(query.text)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top scores > 0
        top_indices = [i for i, score in enumerate(doc_scores) if score > 0]
        top_indices = sorted(top_indices, key=lambda i: doc_scores[i], reverse=True)[:limit]
        
        results = []
        for i in top_indices:
            node = Node(id=self.doc_ids[i], text=self.corpus[i], metadata=self.metadatas[i], score=float(doc_scores[i]))
            results.append(RetrievalResult(node=node, score=float(doc_scores[i])))
            
        return results

class BM25Store(BM25Retriever):
    """Legacy BM25Store maintaining old API while using new contracts internally."""
    
    def __init__(self, persist_path: str = "bm25_index.pkl"):
        super().__init__()
        self.persist_path = persist_path
        self._load()

    def _load(self):
        import pickle
        import os
        if os.path.exists(self.persist_path):
            print(f"Loading BM25 index from {self.persist_path}...")
            try:
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.corpus = data.get("corpus", [])
                    self.doc_ids = data.get("doc_ids", [])
                    self.metadatas = data.get("metadatas", [])
                if self.corpus:
                    tokenized_corpus = [tokenize(doc) for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                self._is_synced = True
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")
                self._is_synced = False
        else:
            self._is_synced = True

    def _save(self):
        import pickle
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "corpus": self.corpus,
                    "doc_ids": self.doc_ids,
                    "metadatas": self.metadatas
                }, f)
        except Exception as e:
            print(f"Failed to save BM25 index: {e}")

    def sync_from_qdrant(self, vector_store):
        """DEPRECATED: Now uses independent disk persistence."""
        pass

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
        self._save()

    def search(self, query: str, limit: int = 5):
        """Legacy search returning dicts."""
        q = Query(text=query)
        results = self.retrieve(q, limit=limit)
        return [{"score": r.score, "payload": {"text": r.node.text, **r.node.metadata}, "id": r.node.id} for r in results]

# -------------------------------------------------------------
# PHASE 4: Component Registration
# -------------------------------------------------------------

def create_bm25_retriever() -> BaseRetriever:
    return BM25Store()



def register(registry):
    registry.register_retriever("bm25", create_bm25_retriever)

from sentence_transformers import SentenceTransformer
from core.config import settings
from core.contracts import BaseEmbeddingProvider
from typing import List

import threading

class EmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self.model is None:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self.model

    def embed_text(self, text: str):
        return self._get_model().encode(text).tolist()

    def embed_texts(self, texts: list[str]):
        return self._get_model().encode(texts).tolist()

    @property
    def vector_size(self):
        model_name = settings.EMBEDDING_MODEL.lower()
        if "minilm" in model_name:
            return 384
        if "mpnet" in model_name:
            return 768

        m = self._get_model()
        if hasattr(m, 'get_embedding_dimension'):
            return m.get_embedding_dimension()
        return m.get_sentence_embedding_dimension()

# -------------------------------------------------------------
# PHASE 4: Component Registration
# -------------------------------------------------------------

def create_embedding_provider() -> BaseEmbeddingProvider:
    return EmbeddingProvider()


def register(registry):
    registry.register_embedding_provider("sentence_transformers", create_embedding_provider)

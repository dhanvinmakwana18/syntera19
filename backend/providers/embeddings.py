from sentence_transformers import SentenceTransformer
from core.config import settings

class EmbeddingProvider:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed_text(self, text: str):
        return self.model.encode(text).tolist()

    def embed_texts(self, texts: list[str]):
        return self.model.encode(texts).tolist()

    @property
    def vector_size(self):
        if hasattr(self.model, 'get_embedding_dimension'):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

embedding_provider = EmbeddingProvider()

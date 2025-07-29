# embedding/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_many(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    
    def embed_one(self, text):
        return self.model.encode([text])[0]

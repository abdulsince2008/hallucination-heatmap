"""Embedding utilities using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class Embedder:
    """Wrapper around sentence-transformers for consistent embedding."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returning normalized embeddings."""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]
    
    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two normalized vectors."""
        # Since embeddings are normalized, cosine similarity = dot product
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(a, b)
        return float(1.0 - similarity)
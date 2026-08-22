from __future__ import annotations

import numpy as np


class EmbeddingModel:
    """Lazy SentenceTransformer wrapper for document and query embeddings."""

    def __init__(
        self,
        model_name: str,
        expected_dim: int = 384,
        query_prefix: str = "",
    ):
        self.model_name = model_name
        self.expected_dim = expected_dim
        self.query_prefix = query_prefix
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def _validate_dim(self, array: np.ndarray) -> np.ndarray:
        if array.ndim == 1:
            dim = array.shape[0]
        else:
            dim = array.shape[1]
        if dim != self.expected_dim:
            raise ValueError(
                f"Embedding model returned dimension {dim}, but the database schema "
                f"expects {self.expected_dim}. Update EMBEDDING_DIM and schema together."
            )
        return array.astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validate_dim(np.asarray(vectors))

    def embed_query(self, query: str) -> np.ndarray:
        retrieval_query = f"{self.query_prefix}{query}" if self.query_prefix else query
        vector = self.model.encode(
            retrieval_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validate_dim(np.asarray(vector))

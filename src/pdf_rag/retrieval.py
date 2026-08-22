from __future__ import annotations

from .database import Database
from .embeddings import EmbeddingModel
from .fusion import reciprocal_rank_fusion
from .models import SearchResult



class HybridRetriever:
    def __init__(
        self,
        database: Database,
        embedder: EmbeddingModel,
        *,
        dense_k: int = 20,
        lexical_k: int = 20,
        fused_k: int = 25,
        rrf_k: int = 60,
    ):
        self.database = database
        self.embedder = embedder
        self.dense_k = dense_k
        self.lexical_k = lexical_k
        self.fused_k = fused_k
        self.rrf_k = rrf_k

    def dense(self, query: str, limit: int | None = None) -> list[SearchResult]:
        embedding = self.embedder.embed_query(query)
        return self.database.dense_search(embedding, limit or self.dense_k)

    def lexical(self, query: str, limit: int | None = None) -> list[SearchResult]:
        return self.database.lexical_search(query, limit or self.lexical_k)

    def hybrid(self, query: str) -> list[SearchResult]:
        embedding = self.embedder.embed_query(query)
        dense_results = self.database.dense_search(embedding, self.dense_k)
        lexical_results = self.database.lexical_search(query, self.lexical_k)
        return reciprocal_rank_fusion(
            [dense_results, lexical_results],
            rrf_k=self.rrf_k,
            limit=self.fused_k,
        )

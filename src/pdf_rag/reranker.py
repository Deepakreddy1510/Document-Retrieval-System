from __future__ import annotations

from copy import deepcopy

from .models import SearchResult


class Reranker:
    """Lazy CrossEncoder reranker applied only to first-stage candidates."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked: list[SearchResult] = []
        for candidate, score in zip(candidates, scores, strict=True):
            item = deepcopy(candidate)
            item.rerank_score = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)
        return reranked[:top_k]

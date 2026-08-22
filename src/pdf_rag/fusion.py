from __future__ import annotations

from copy import deepcopy

from .models import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    *,
    rrf_k: int = 60,
    limit: int = 25,
) -> list[SearchResult]:
    """Fuse ranked lists with RRF without mixing incomparable raw score scales."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    combined: dict[int, SearchResult] = {}
    scores: dict[int, float] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            if result.chunk_id not in combined:
                combined[result.chunk_id] = deepcopy(result)
                scores[result.chunk_id] = 0.0
            else:
                existing = combined[result.chunk_id]
                if existing.dense_score is None and result.dense_score is not None:
                    existing.dense_score = result.dense_score
                if existing.lexical_score is None and result.lexical_score is not None:
                    existing.lexical_score = result.lexical_score
            scores[result.chunk_id] += 1.0 / (rrf_k + rank)

    fused = []
    for chunk_id, result in combined.items():
        result.rrf_score = scores[chunk_id]
        fused.append(result)

    fused.sort(key=lambda item: item.rrf_score or 0.0, reverse=True)
    return fused[:limit]

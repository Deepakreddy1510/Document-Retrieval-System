from __future__ import annotations

from dataclasses import dataclass

from .models import SearchResult


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def relevant_keys(expected_sources: list[dict]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for source in expected_sources:
        document = source["document"]
        for page in source.get("pages", []):
            keys.add((document, int(page)))
    return keys


def score_retrieval(
    results: list[SearchResult],
    expected_sources: list[dict],
    *,
    k: int,
) -> RetrievalMetrics:
    expected = relevant_keys(expected_sources)
    top = results[:k]

    if not expected:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0)

    retrieved_keys = [(r.filename, r.page_number) for r in top]
    relevant_retrieved = {key for key in retrieved_keys if key in expected}

    hit = 1.0 if relevant_retrieved else 0.0
    precision = len([key for key in retrieved_keys if key in expected]) / max(k, 1)
    recall = len(relevant_retrieved) / len(expected)

    reciprocal_rank = 0.0
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in expected:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalMetrics(hit, precision, recall, reciprocal_rank)

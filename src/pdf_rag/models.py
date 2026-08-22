from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int
    text: str


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    text: str
    dense_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class Source:
    label: str
    chunk_id: int
    filename: str
    page_number: int
    text: str
    score: float | None = None


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    retrieval_query: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    filename: str
    document_id: int | None
    page_count: int
    chunk_count: int
    status: str

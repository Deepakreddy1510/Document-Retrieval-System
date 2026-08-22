from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def _default_model(provider: str) -> str:
    if provider == "gemini":
        return "gemini-3.7-flash"
    return "llama-3.3-70b-versatile"


_PROVIDER = _provider()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://rag:rag@localhost:5432/ragdb",
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    embedding_query_prefix: str = os.getenv(
        "EMBEDDING_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    llm_provider: str = _PROVIDER
    llm_model: str = os.getenv("LLM_MODEL") or _default_model(_PROVIDER)
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "450"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))

    dense_k: int = int(os.getenv("DENSE_K", "20"))
    lexical_k: int = int(os.getenv("LEXICAL_K", "20"))
    fused_k: int = int(os.getenv("FUSED_K", "25"))
    final_k: int = int(os.getenv("FINAL_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))


settings = Settings()

from __future__ import annotations

import hashlib

from .config import Settings, settings
from .context import build_context
from .database import Database
from .embeddings import EmbeddingModel
from .generation import LLMGenerator
from .models import AnswerResult, IngestionResult, SearchResult
from .pdf_processor import extract_pdf_bytes
from .reranker import Reranker
from .retrieval import HybridRetriever
from .chunking import chunk_pages


class RAGPipeline:
    def __init__(self, config: Settings = settings):
        self.config = config
        self.database = Database(config.database_url)
        self.embedder = EmbeddingModel(
            config.embedding_model,
            expected_dim=config.embedding_dim,
            query_prefix=config.embedding_query_prefix,
        )
        self.retriever = HybridRetriever(
            self.database,
            self.embedder,
            dense_k=config.dense_k,
            lexical_k=config.lexical_k,
            fused_k=config.fused_k,
            rrf_k=config.rrf_k,
        )
        self.reranker = Reranker(config.reranker_model)
        self.generator = LLMGenerator(
            provider=config.llm_provider,
            model=config.llm_model,
            gemini_api_key=config.gemini_api_key,
            groq_api_key=config.groq_api_key,
        )

    def initialize(self) -> None:
        self.database.initialize_schema()

    def ingest_pdf(self, *, filename: str, pdf_bytes: bytes) -> IngestionResult:
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        existing = self.database.find_document_by_hash(file_hash)
        if existing:
            return IngestionResult(
                filename=existing["filename"],
                document_id=int(existing["id"]),
                page_count=int(existing["page_count"]),
                chunk_count=0,
                status="already_indexed",
            )

        pages = extract_pdf_bytes(pdf_bytes)
        tokenizer = self.embedder.tokenizer
        model_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_limit, int) and model_limit < 1_000_000:
            # Leave room for model-added special tokens during embedding.
            safe_limit = max(1, model_limit - 2)
            if self.config.chunk_size > safe_limit:
                raise ValueError(
                    f"CHUNK_SIZE={self.config.chunk_size} exceeds the embedding model's "
                    f"safe token limit ({safe_limit}). Reduce CHUNK_SIZE to avoid truncation."
                )

        chunks = chunk_pages(
            pages,
            tokenizer,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            raise ValueError("The PDF produced no non-empty chunks.")

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        document_id = self.database.store_document_with_chunks(
            filename=filename,
            file_hash=file_hash,
            page_count=len(pages),
            chunks=chunks,
            embeddings=embeddings,
        )
        return IngestionResult(
            filename=filename,
            document_id=document_id,
            page_count=len(pages),
            chunk_count=len(chunks),
            status="indexed",
        )

    def retrieve(self, question: str) -> list[SearchResult]:
        # V1 production retrieval:
        # Dense retrieval was selected through offline ablation testing.
        # Hybrid lexical + RRF retrieval is kept in the codebase for experiments.
        candidates = self.retriever.dense(
            question,
            limit=max(self.config.dense_k, self.config.final_k),
        )

        return self.reranker.rerank(
            question,
            candidates,
            top_k=self.config.final_k,
        )

    def answer(self, question: str) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        final_results = self.retrieve(question)
        context, sources = build_context(
            final_results,
            max_chars=self.config.max_context_chars,
        )
        answer = self.generator.generate(question=question, context=context)
        return AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
            # v1 intentionally uses the original query. Contextual rewriting is an
            # optional experiment to add only after baseline retrieval evaluation.
            retrieval_query=question,
        )

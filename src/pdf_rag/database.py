from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .models import Chunk, SearchResult


class Database:
    def __init__(self, database_url: str):
        self.database_url = database_url

    @contextmanager
    def connection(self, *, register_pgvector: bool = True):
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            if register_pgvector:
                register_vector(conn)
            yield conn

    def initialize_schema(self) -> None:
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        # vector must exist before pgvector's Python type can be registered.
        with psycopg.connect(self.database_url) as conn:
            conn.execute(schema_sql)

    def healthcheck(self) -> bool:
        with self.connection() as conn:
            return conn.execute("SELECT 1 AS ok").fetchone()["ok"] == 1

    def find_document_by_hash(self, file_hash: str) -> dict | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT id, filename, page_count FROM documents WHERE file_hash = %s",
                (file_hash,),
            ).fetchone()

    def store_document_with_chunks(
        self,
        *,
        filename: str,
        file_hash: str,
        page_count: int,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        with self.connection() as conn:
            document = conn.execute(
                """
                INSERT INTO documents (filename, file_hash, page_count)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (filename, file_hash, page_count),
            ).fetchone()
            document_id = int(document["id"])

            rows: Iterable[tuple] = (
                (
                    document_id,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                    Vector(embedding),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (
                        document_id, page_number, chunk_index, text, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            return document_id

    def list_documents(self) -> list[dict]:
        with self.connection() as conn:
            return list(
                conn.execute(
                    """
                    SELECT d.id, d.filename, d.page_count, d.created_at,
                           COUNT(c.id)::int AS chunk_count
                    FROM documents d
                    LEFT JOIN chunks c ON c.document_id = d.id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                    """
                ).fetchall()
            )

    def count_chunks(self) -> int:
        with self.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    @staticmethod
    def _row_to_result(row: dict) -> SearchResult:
        return SearchResult(
            chunk_id=int(row["chunk_id"]),
            document_id=int(row["document_id"]),
            filename=row["filename"],
            page_number=int(row["page_number"]),
            chunk_index=int(row["chunk_index"]),
            text=row["text"],
            dense_score=float(row["dense_score"]) if row.get("dense_score") is not None else None,
            lexical_score=float(row["lexical_score"]) if row.get("lexical_score") is not None else None,
        )

    def dense_search(self, query_embedding: np.ndarray, limit: int = 20) -> list[SearchResult]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.page_number,
                    c.chunk_index,
                    c.text,
                    1 - (c.embedding <=> %s) AS dense_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (Vector(query_embedding), Vector(query_embedding), limit),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def lexical_search(self, query: str, limit: int = 20) -> list[SearchResult]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                WITH q AS (
                    SELECT websearch_to_tsquery('english', %s) AS query
                )
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.page_number,
                    c.chunk_index,
                    c.text,
                    ts_rank_cd(c.text_search, q.query) AS lexical_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE c.text_search @@ q.query
                ORDER BY lexical_score DESC
                LIMIT %s
                """,
                (query, limit),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

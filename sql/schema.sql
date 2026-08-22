CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL CHECK (page_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number >= 1),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    text_search TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(text, ''))
    ) STORED,
    UNIQUE (document_id, page_number, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_text_search
    ON chunks USING GIN(text_search);

-- Deliberately no HNSW/IVFFlat index in v1.
-- For the small placement-project corpus we start with exact cosine search and
-- add ANN indexing only after a measured latency/recall experiment justifies it.

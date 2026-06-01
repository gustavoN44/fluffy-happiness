-- Phase 1 schema: a single table holding document chunks and their embeddings.
--
-- Runs automatically on a fresh volume (after 01-create-extension.sql, which
-- the lexical ordering guarantees). On an already-initialized volume it must
-- be applied manually; IF NOT EXISTS makes that safe to re-run.

CREATE TABLE IF NOT EXISTS chunks (
    id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content   TEXT         NOT NULL,

    -- Dimensionality MUST equal the embedding model's output. OpenAI
    -- text-embedding-3-small emits exactly 1536 dimensions; a mismatch makes
    -- every insert fail. Changing embedding models later means changing this.
    embedding vector(1536) NOT NULL,

    -- Unused in Phase 1, added now to avoid a later migration. Will hold
    -- document source, chunk index, and (Phase 4) RBAC ownership fields.
    metadata  JSONB        NOT NULL DEFAULT '{}'::jsonb
);

-- Intentionally NO vector index (HNSW/IVFFlat) yet. With one document and a
-- handful of chunks, a brute-force exact scan is faster and simpler than an
-- approximate index, and avoids tuning we don't need. Add one when the corpus
-- grows enough that scan latency actually shows up in the eval metrics.

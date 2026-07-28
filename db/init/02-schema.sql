-- Phase 3 schema: the config registry.
--
-- Chunk tables are NO LONGER static. Each RunConfig gets its own table
-- `chunks_<config_id>` with a vector(dim) matching its embedder, created by the
-- app at ingest time (dimension is only known from the config). See app/db.py
-- (ensure_config_table) and DECISIONS.md D6.
--
-- This registry records what each config_id is. Runs automatically on a fresh
-- volume; app/db.ensure_registry() creates it idempotently on an existing one.

CREATE TABLE IF NOT EXISTS configs (
    config_id       TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    chunker         TEXT NOT NULL,
    chunker_params  JSONB NOT NULL,
    embedder        TEXT NOT NULL,
    embedder_params JSONB NOT NULL,
    dim             INT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable pgvector. Runs once, automatically, when the database is first
-- initialized on an empty data volume. IF NOT EXISTS makes it safe to re-run
-- manually (e.g. psql) without erroring.
CREATE EXTENSION IF NOT EXISTS vector;
